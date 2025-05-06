#!/bin/bash

LOG_FILE="pipeline.log"
echo "=== Запуск пайплайна $(date) ===" | tee $LOG_FILE

echo "=== Установка зависимостей ===" | tee -a $LOG_FILE
pip install pandas scikit-learn joblib matplotlib 2>&1 | tee -a $LOG_FILE

echo "=== Генерация данных ===" | tee -a $LOG_FILE
python data_creation.py 2>&1 | tee -a $LOG_FILE

echo "=== Предобработка данных ===" | tee -a $LOG_FILE
python model_preparation.py 2>&1 | tee -a $LOG_FILE

echo "=== Обучение модели ===" | tee -a $LOG_FILE
python model_preprocessing.py 2>&1 | tee -a $LOG_FILE

echo "=== Оценка модели ===" | tee -a $LOG_FILE
python model_testing.py 2>&1 | tee -a $LOG_FILE

echo "=== Пайплайн завершён $(date) ===" | tee -a $LOG_FILE
