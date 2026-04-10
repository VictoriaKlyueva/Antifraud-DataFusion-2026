# 🛡️ Antifraud Detection - DataFusion 2026

**45/574 место на лидерборде** | Хакатон DataFusion 2026

## 📋 Описание задачи

Детекция мошеннических транзакций в банковских данных. Задача заключалась в построении модели для идентификации фродовых операций на основе истории транзакций клиентов.

### Метрика
- **PR-AUC** (Precision-Recall Area Under Curve)
- Оптимален для несбалансированных данных (fraud rate < 1%)

### Данные
- **Pretrain**: История транзакций без меток (для предобучения)
- **Train**: Размеченные транзакции с метками fraud/legit
- **Test**: Транзакции для предсказания

---

## 🏗️ Архитектура решения

### Подход 1: CatBoost + Feature Engineering

#### Feature Engineering

**1. Кумулятивные признаки**
```python
- cum_count: Количество операций до текущей
- cum_sum: Накопленная сумма транзакций
- cum_avg: Средняя сумма транзакций
- amount_diff_from_cum_avg: Отклонение от среднего
```

**2. Временные паттерны**
```python
- ops_7d, ops_14d, ops_21d: Количество операций за N дней
- avg_amt_7d, avg_amt_14d, avg_amt_21d: Средняя сумма за N дней
- hour_of_day: Час совершения операции
- is_night: Флаг ночной операции (23:00-06:00)
- night_ops_ratio: Доля ночных операций в истории
```

**3. Device Intelligence**
```python
- device_risk_score: Композитный риск устройства
  (compromised * 3 + web_rdp_connection * 2 + phone_voip_call_state * 1)
- timezone_changed: Смена временной зоны
- browser_language_changes: Количество смен языка браузера
```

**4. Агрегатные фичи по клиенту**
```python
- customer_total_txns: Общее количество транзакций
- customer_avg_amt: Средняя сумма транзакций клиента
- customer_std_amt: Стандартное отклонение сумм
- customer_unique_mcc: Количество уникальных MCC кодов
- customer_txns_per_hour: Интенсивность транзакций
- amt_zscore: Z-score суммы относительно истории клиента
```

#### Оптимизация гиперпараметров

Использован **Optuna** с TPESampler для подбора параметров:

```python
best_params = {
    'iterations': 3000,
    'learning_rate': 0.0407,
    'depth': 6,
    'l2_leaf_reg': 4.75,
    'subsample': 0.91,
    'colsample_bylevel': 0.87,
    'border_count': 244,
    'bootstrap_type': 'Bernoulli',
    'eval_metric': 'PRAUC',
    'auto_class_weights': 'Balanced',
    'early_stopping_rounds': 100
}
```

**Результаты CatBoost:**
- Validation PR-AUC: **0.8060**
- Best iteration: ~1500

---

### Подход 2: Имплементация FraudTransformer из оригинальной статьи

#### Архитектура модели

```
FraudTransformer
├── Categorical Embeddings (MCC, event_type, currency, etc.)
├── Quantized Numerical Embeddings (amount, time features)
├── Sinusoidal Time Encoder (интервалы между транзакциями)
├── Rotary Positional Embedding (RoPE)
├── Transformer Decoder (6 layers, 8 heads)
├── Aggregate Features Projection
└── Classification Head
```

#### Ключевые компоненты

**1. Rotary Positional Embedding (RoPE)**
- Кодирование позиции транзакции в последовательности
- Лучше сохраняет относительные позиции чем стандартный positional encoding

**2. Sinusoidal Time Encoder**
- Кодирование временных интервалов между транзакциями
- Learnable scaling для адаптации к данным
- Учитывает паттерны активности клиента

**3. Квантизация числовых признаков**
- KBinsDiscretizer с 50 бинами
- Преобразование непрерывных значений в дискретные токены
- Позволяет модели учить нелинейные зависимости

**4. Causal Masking**
- Авторегрессионное моделирование
- Токен может видеть только предыдущие транзакции
- Имитирует реальный сценарий детекции в реальном времени

#### Гиперпараметры

```python
TRANSFORMER_EMBEDDING_DIM = 128
TRANSFORMER_NUM_HEADS = 8
TRANSFORMER_NUM_LAYERS = 6
TRANSFORMER_MAX_SEQ_LEN = 100
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_LEARNING_RATE = 3e-5
TRANSFORMER_BATCH_SIZE = 128
TRANSFORMER_NUM_EPOCHS = 8
```

---

## 📊 Результаты

| Метрика | Validation | Leaderboard |
|---------|------------|-------------|
| PR-AUC | **0.85** | **0.14** |
| Место | - | **46** |

---

## 🔧 Технический стек

- **Python 3.10+**
- **Polars** - Быстрая обработка данных
- **CatBoost** - Градиентный бустинг
- **PyTorch** - Нейронные сети
- **Optuna** - Оптимизация гиперпараметров
- **MLflow** - Трекинг экспериментов
- **ydata-profiling** - EDA и анализ данных

---

## 🚀 Запуск

### Установка зависимостей

```bash
pip install polars catboost torch optuna mlflow ydata-profiling pandas numpy scikit-learn matplotlib seaborn
```

### Данные

Данные должны быть размещены в структуре:
```
data/
├── pretrain/           # Папка с parquet файлами pretrain
├── train/              # Папка с parquet файлами train
├── train_labels.parquet
├── pretest.parquet
├── test.parquet
└── sample_submit.csv
```