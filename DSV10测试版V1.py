# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations, islice
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
    from tensorflow.keras.callbacks import Callback, EarlyStopping
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Input, Dropout
    from keras.callbacks import Callback, EarlyStopping
    from keras.optimizers import Adam
from collections import defaultdict
from tqdm import tqdm
import os
import glob
import joblib
from colorama import Fore, Style, Back

# ================== 用户配置区 ==================
DEBUG_MODE = False            # 调试模式(True跳过组合生成)
HISTORY_WINDOW = 5           # 时间窗口
MIN_HAMMING_DIST = 3         # 最小汉明距离
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]  # 质数定义
PROGRESS_THEME = {           # 增强进度条主题
    "colors": {
        "data": Fore.CYAN,
        "train": Fore.MAGENTA,
        "lstm": Fore.YELLOW,
        "comb": Fore.GREEN,
        "result": Fore.RED
    },
    "icons": {
        "data": "📂",
        "train": "🎓", 
        "lstm": "🧠",
        "comb": "🔀",
        "result": "🏅"
    },
    "bars": {
        "data": "█",
        "train": "▓",
        "lstm": "▒",
        "comb": "░"
    }
}
DATA_DIR = "E:/SSQiu/date/"        # 数据存放目录
MODEL_DIR = "E:/SSQiu/Models_03-25/"     # 模型保存目录
# ===============================================

# 禁用冗余警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

class TqdmEpochCallback(Callback):
    """自定义Keras训练进度条（修复提前停止问题）"""
    def __init__(self, total_epochs):
        super().__init__()
        self.epoch_bar = None
        self.total_epochs = total_epochs
        
    def on_train_begin(self, logs=None):
        self.epoch_bar = tqdm(total=self.total_epochs, 
                             desc=f"{PROGRESS_THEME['icons']['lstm']} LSTM训练进度",
                             bar_format=f"{PROGRESS_THEME['colors']['lstm']}{{l_bar}}{{bar}}{Style.RESET_ALL}",
                             colour='GREEN')

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
        val_loss = logs.get('val_loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        self.epoch_bar.set_postfix_str(f"loss: {logs.get('loss', 0):.4f} | acc: {logs.get('accuracy', 0):.2f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.2f}")

class LotteryPredictor:
    def __init__(self):
        self.red_models = {}
        self.blue_models = {}
        self.lstm_model = None
        self.history_vectors = []
        self.last_trained_date = None

    def load_historical_data(self):
        """安全加载历史数据（增强数据清洗）"""
        print(f"\n{PROGRESS_THEME['icons']['data']} 加载历史数据...")
        all_files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
        dfs = []
        
        with tqdm(all_files, desc=f"{PROGRESS_THEME['icons']['data']} 读取数据文件",
                 bar_format=f"{PROGRESS_THEME['colors']['data']}{{l_bar}}{{bar}}{Style.RESET_ALL}") as pbar:
            for f in pbar:
                df = pd.read_excel(f)
                # 强化数据清洗：处理异常值和格式问题
                for col in [f'红球{i}' for i in range(1,7)] + ['蓝球']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # 过滤超出范围的数值并随机填充
                    if '红球' in col:
                        df[col] = df[col].apply(lambda x: x if 1<=x<=33 else np.random.randint(1,34))
                    else:
                        df[col] = df[col].apply(lambda x: x if 1<=x<=16 else np.random.randint(1,17))
                    df[col] = df[col].astype(int)
                dfs.append(df)
                pbar.set_postfix_str(f"已加载 {os.path.basename(f)}")

        full_df = pd.concat(dfs).sort_values('开奖日期').drop_duplicates().reset_index(drop=True)
        print(f"{PROGRESS_THEME['colors']['data']}✅ 共加载 {len(full_df)} 期历史数据{Style.RESET_ALL}")
        return full_df

    def calculate_combination_prob(self, red_combo, red_probs, blue_prob, last_3_reds):
        """计算组合概率（增加连号奖励机制）"""
        prob_array = np.array([red_probs[num] for num in red_combo])
        red_prob = np.prod(prob_array)
        
        # 连号奖励机制：每出现一个连号增加5%概率
        sorted_combo = sorted(red_combo)
        consecutive_count = sum(1 for i in range(len(sorted_combo)-1) if sorted_combo[i+1] - sorted_combo[i] == 1)
        consecutive_bonus = 1.05 ** consecutive_count
        
        # 重复号码惩罚机制
        repeat_penalty = 0.95 ** len(set(red_combo) & last_3_reds)
        
        return red_prob * blue_prob * repeat_penalty * consecutive_bonus

    def extract_features(self, red_balls, blue_balls):
        """特征工程（新增连号组数特征）"""
        features = []
        if len(red_balls) < HISTORY_WINDOW + 1:
            raise ValueError(f"需要至少{HISTORY_WINDOW+1}期数据")

        columns = [
            'consecutive', 'consecutive_groups', 'odd', 'sum', 'large', 'prime',
            'tails', 'max_gap', 'tail_dist'  # 新增连号组数特征
        ] + [f'hist_{j}_red_overlap' for j in range(1, HISTORY_WINDOW+1)] + \
            [f'hist_{j}_blue_match' for j in range(1, HISTORY_WINDOW+1)]

        with tqdm(total=len(red_balls)-HISTORY_WINDOW, 
                 desc=f"{PROGRESS_THEME['icons']['data']} 特征工程",
                 bar_format=f"{PROGRESS_THEME['colors']['data']}{{l_bar}}{{bar}}{Style.RESET_ALL}") as main_bar:
            for i in range(HISTORY_WINDOW, len(red_balls)):
                red = red_balls[i]
                sorted_red = sorted(red)
                
                # 特征计算带子进度（总特征数+1）
                with tqdm(total=9, desc="特征计算", leave=False) as sub_bar:  
                    diffs = np.diff(sorted_red)
                    consecutive = np.sum(diffs == 1)
                    sub_bar.update(1)
                    
                    # 计算连号组数（新增特征）
                    consecutive_groups = 0
                    temp_count = 0
                    for d in diffs:
                        if d == 1:
                            temp_count += 1
                            if temp_count == 1:  # 新连号组开始
                                consecutive_groups += 1
                        else:
                            temp_count = 0
                    sub_bar.update(1)
                    
                    odd_count = np.sum(np.array(sorted_red) % 2 != 0)
                    sub_bar.update(1)
                    
                    sum_red = np.sum(sorted_red)
                    sub_bar.update(1)
                    
                    large_count = np.sum(np.array(sorted_red) > 16)
                    sub_bar.update(1)
                    
                    prime_count = len(np.intersect1d(sorted_red, PRIMES))
                    sub_bar.update(1)
                    
                    tails = np.array(sorted_red) % 10
                    unique_tails = len(np.unique(tails))
                    sub_bar.update(1)
                    
                    max_gap = np.max(diffs)
                    sub_bar.update(1)
                    
                    tail_dist = np.sum(np.bincount(tails))
                    sub_bar.update(1)

                # 时间特征处理
                time_features = []
                with tqdm(total=HISTORY_WINDOW*2, desc="时间特征", leave=False) as time_bar:
                    for j in range(1, HISTORY_WINDOW+1):
                        time_features.append(len(np.intersect1d(red, red_balls[i-j])))
                        time_bar.update(1)
                        time_features.append(int(blue_balls[i] == blue_balls[i-j]))
                        time_bar.update(1)

                features.append([
                    consecutive, consecutive_groups, odd_count, sum_red, large_count, prime_count,
                    unique_tails, max_gap, tail_dist, *time_features
                ])
                main_bar.update(1)
        
        return pd.DataFrame(features, columns=columns)

    def build_lstm_model(self):
        """LSTM模型（优化网络结构防止过拟合）"""
        model_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        if os.path.exists(model_path):
            print(f"{PROGRESS_THEME['colors']['lstm']}🔁 加载现有LSTM模型{Style.RESET_ALL}")
            return load_model(model_path)
            
        model = Sequential([
            Input(shape=(HISTORY_WINDOW, 6)),
            LSTM(64, return_sequences=True, dropout=0.2),  # 减少神经元数量
            Dropout(0.3),
            LSTM(32, dropout=0.2),
            Dense(33, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.0005),  # 降低学习率
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_models(self, features_df, y_red, y_blue):
        """优化后的训练流程（增加类别平衡）"""
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        print(f"\n{PROGRESS_THEME['colors']['train']}{PROGRESS_THEME['icons']['train']} 模型训练{Style.RESET_ALL}")
        with tqdm(total=33+16, 
                 desc=f"{PROGRESS_THEME['icons']['train']} 模型训练进度",
                 bar_format=f"{PROGRESS_THEME['colors']['train']}{{l_bar}}{{bar}}{Style.RESET_ALL}") as pbar:
                 
            # 红球模型训练（动态类别权重）
            for num in range(1, 34):
                model_path = os.path.join(MODEL_DIR, f"red_{num}.pkl")
                if os.path.exists(model_path):
                    self.red_models[num] = joblib.load(model_path)
                else:
                    y = [1 if num in row else 0 for row in y_red]
                    class_weight = {0: 1.0, 1: len(y)/sum(y)-1}  # 动态计算权重
                    
                    self.red_models[num] = RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,  # 降低树深
                        class_weight=class_weight,
                        n_jobs=-1,
                        verbose=0
                    )
                    self.red_models[num].fit(features_df, y)
                    joblib.dump(self.red_models[num], model_path)
                pbar.update(1)
                pbar.set_postfix_str(f"红球 {num}/33")

            # 蓝球模型训练（增加正则化）
            for num in range(1, 17):
                model_path = os.path.join(MODEL_DIR, f"blue_{num}.pkl")
                if os.path.exists(model_path):
                    self.blue_models[num] = joblib.load(model_path)
                else:
                    y = [1 if num == b else 0 for b in y_blue]
                    class_weight = {0: 1.0, 1: len(y)/sum(y)-1}
                    
                    self.blue_models[num] = RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,  # 限制树深
                        class_weight=class_weight,
                        n_jobs=-1,
                        verbose=0
                    )
                    self.blue_models[num].fit(features_df, y)
                    joblib.dump(self.blue_models[num], model_path)
                pbar.update(1)
                pbar.set_postfix_str(f"蓝球 {num}/16")

        # LSTM训练（优化训练参数）
        lstm_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        self.lstm_model = self.build_lstm_model()
        if not os.path.exists(lstm_path):
            print(f"{PROGRESS_THEME['colors']['lstm']}{PROGRESS_THEME['icons']['lstm']} LSTM训练{Style.RESET_ALL}")
            red_sequences = [row.tolist() for row in y_red]
            X, y = [], []
            
            for i in range(len(red_sequences) - HISTORY_WINDOW):
                X.append(red_sequences[i:i+HISTORY_WINDOW])
                target = np.zeros(33)
                np.put(target, [num-1 for num in red_sequences[i+HISTORY_WINDOW]], 1)
                y.append(target)
            
            self.lstm_model.fit(np.array(X), np.array(y), 
                              epochs=200,  # 增加总epoch数
                              batch_size=64,  # 增大批次尺寸
                              validation_split=0.2,
                              verbose=0,
                              callbacks=[
                                  TqdmEpochCallback(200),
                                  EarlyStopping(patience=10, restore_best_weights=True)  # 增加耐心值
                              ])
            self.lstm_model.save(lstm_path)

    def generate_predictions(self, latest_features, red_balls):
        """预测结果生成（动态权重调整）"""
        if self.lstm_model is None:
            self.lstm_model = self.build_lstm_model()
            
        red_probs = defaultdict(float)
        blue_probs = defaultdict(float)
        
        # 历史热号计算（滑动窗口机制）
        all_reds = [num for sublist in red_balls[-200:] for num in sublist]  # 扩大统计窗口
        red_counts = defaultdict(int, {k: v for k, v in zip(*np.unique(all_reds, return_counts=True))})
        hot_reds = sorted(red_counts.items(), key=lambda x: x[1], reverse=True)[:15]  # 取前15个热门
        
        # 红球预测（动态权重调整）
        with ThreadPoolExecutor() as executor:
            futures = {num: executor.submit(model.predict_proba, latest_features) 
                     for num, model in self.red_models.items()}
            for num, future in futures.items():
                base_prob = future.result()[0][1]
                # 动态权重：热门号码权重降低，冷门权重提高
                heat_level = next((i for i, (n, _) in enumerate(hot_reds) if n == num), 15)
                weight = 0.5 + 0.3 * (1 - heat_level/15)  # 权重范围0.5-0.8
                red_probs[num] = base_prob * weight

        # 蓝球预测（时序平滑处理）
        with ThreadPoolExecutor() as executor:
            futures = {num: executor.submit(model.predict_proba, latest_features)
                     for num, model in self.blue_models.items()}
            for num, future in futures.items():
                decay_factor = 0.9 ** (len(self.blue_models) - num)  # 号码越大衰减越大
                blue_probs[num] = future.result()[0][1] * decay_factor

        # LSTM预测（加权融合）
        lstm_input = np.array([red_balls[-HISTORY_WINDOW:]])
        lstm_pred = self.lstm_model.predict(lstm_input, verbose=0)[0]
        for num in range(1, 34):
            heat_weight = 0.2 if num in dict(hot_reds) else 0
            red_probs[num] = 0.5*red_probs[num] + 0.3*lstm_pred[num-1] + heat_weight

        return red_probs, blue_probs

    def optimized_combination_generation(self, red_probs, blue_probs, red_balls):
        """优化组合生成（引入多样性机制）"""
        candidate_pool = defaultdict(float)
        last_3_reds = set(red_balls[-3:].flatten())
        
        # 候选池优化（冷热混合+随机采样）
        top_50_red = sorted(red_probs, key=red_probs.get, reverse=True)[:40]  # 前40热号
        cold_reds = sorted(red_probs, key=red_probs.get)[:10]  # 前10冷号
        random_reds = np.random.choice(list(red_probs.keys()), size=5, p=np.array(list(red_probs.values()))/sum(red_probs.values()))  # 概率采样
        top_50_red = list(set(top_50_red + cold_reds + random_reds.tolist()))[:50]
        
        print(f"\n{PROGRESS_THEME['colors']['comb']}{PROGRESS_THEME['icons']['comb']} 智能组合生成{Style.RESET_ALL}")
        
        BATCH_SIZE = 5000
        total_comb = sum(1 for _ in combinations(top_50_red, 6))
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            with tqdm(total=total_comb, 
                     desc=f"{PROGRESS_THEME['icons']['comb']} 生成候选组合",
                     bar_format=f"{PROGRESS_THEME['colors']['comb']}{{l_bar}}{{bar}}{Style.RESET_ALL}") as main_bar:
                batch_combos = combinations(top_50_red, 6)
                
                while True:
                    batch = list(islice(batch_combos, BATCH_SIZE))
                    if not batch:
                        break
                    
                    # 向量化处理（优化内存）
                    vectors = np.zeros((len(batch), 33), dtype=np.int8)
                    for i, comb in enumerate(batch):
                        vectors[i, [num-1 for num in comb]] = 1
                    main_bar.update(len(batch))

                    # 距离验证（矩阵运算优化）
                    if self.history_vectors:
                        ref_vectors = np.array(self.history_vectors[-500:])  # 仅比较最近500期
                        distances = np.sum(np.bitwise_xor(vectors[:, None], ref_vectors), axis=2) / 33
                        valid_mask = np.all(distances >= MIN_HAMMING_DIST/33, axis=1)
                        valid_batch = [batch[i] for i in np.where(valid_mask)[0]]
                    else:
                        valid_batch = batch
                    
                    # 并行任务提交（优化蓝球权重）
                    futures = []
                    for comb in valid_batch:
                        for blue in range(1,17):
                            adj_blue_prob = blue_probs[blue] * (0.95 ** abs(blue-8))  # 偏好中间号码
                            future = executor.submit(
                                self.calculate_combination_prob,
                                comb, red_probs, adj_blue_prob, last_3_reds
                            )
                            futures.append( (future, comb, blue) )
                    
                    # 结果收集与归一化
                    total_prob = 0
                    for future, comb, blue in futures:
                        try:
                            prob = future.result()
                            candidate_pool[(tuple(sorted(comb)), blue)] = prob
                            total_prob += prob
                        except Exception as e:
                            print(f"\n计算异常：{str(e)}")
                    
                    # 概率归一化处理
                    if total_prob > 0:
                        for key in candidate_pool:
                            candidate_pool[key] /= total_prob

        return candidate_pool

if __name__ == "__main__":
    # 初始化预测器
    predictor = LotteryPredictor()
    
    try:
        # 增量数据加载（增强校验）
        full_data = predictor.load_historical_data()
        
        # 数据格式转换（范围校验）
        red_cols = [f'红球{i}' for i in range(1,7)]
        for col in red_cols + ['蓝球']:
            full_data[col] = full_data[col].apply(
    lambda x: x if ( (1<=x<=33) if '红球' in col else (1<=x<=16) ) else np.random.randint(1,34 if '红球' in col else 17)
)
        
        red_balls = full_data[red_cols].values
        blue_balls = full_data['蓝球'].values
        
        # 特征工程（异常处理）
        try:
            features_df = predictor.extract_features(red_balls, blue_balls)
        except ValueError as e:
            print(f"{Fore.RED}错误：{str(e)}，请检查数据完整性！{Style.RESET_ALL}")
            exit()
        
        y_red = red_balls[HISTORY_WINDOW:]
        y_blue = blue_balls[HISTORY_WINDOW:]
        
        # 模型训练（完整流程）
        predictor.train_models(features_df, y_red, y_blue)
        
        # 生成预测（使用最新3期数据）
        latest_features = predictor.extract_features(red_balls[-HISTORY_WINDOW-3:], 
                                                   blue_balls[-HISTORY_WINDOW-3:])
        red_probs, blue_probs = predictor.generate_predictions(latest_features.iloc[[-1]], red_balls)
        
        # 组合生成（调试模式跳过）
        if DEBUG_MODE:
            print(f"\n{PROGRESS_THEME['colors']['result']}🔧 调试模式：跳过组合生成{Style.RESET_ALL}")
            candidate_pool = {((1,2,3,4,5,6), 1): 0.001}
        else:
            candidate_pool = predictor.optimized_combination_generation(red_probs, blue_probs, red_balls)
        
        # 结果展示（概率分布分析）
        print(f"\n{PROGRESS_THEME['colors']['result']}{'━'*30} 预测结果 {'━'*30}{Style.RESET_ALL}")
        top_combinations = sorted(candidate_pool.items(), key=lambda x: x[1], reverse=True)[:5]
        total_prob = sum(v for _, v in top_combinations)
        
        for rank, ((red, blue), prob) in enumerate(top_combinations):
            color = PROGRESS_THEME['colors']['result']
            print(f"\n{color}🏆 第{rank+1}推荐组合 | 占比: {prob/total_prob:.2%}{Style.RESET_ALL}")
            print(f"{color}├─ 🔴 红球：{sorted(red)}")
            print(f"{color}└─ 🔵 蓝球：{blue} | 综合概率：{prob:.8f}{Style.RESET_ALL}")
            print(f"{color}{'─'*68}{Style.RESET_ALL}")

    except Exception as e:
        print(f"\n{Fore.RED}⚠️ 运行出错：{str(e)}{Style.RESET_ALL}")