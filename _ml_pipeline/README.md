# ML Pipeline

## Запуск
_На этапе PoC процесс полуавтоматический._

### Перед запуском пайплайна
Необходимо вручную скопировать исходный датасет в _ml_pipeline/data/raw
Состав исходного датасета JailbreakDB (https://huggingface.co/datasets/youbin2014/JailbreakDB):
- text_jailbreak_unique.csv (https://huggingface.co/datasets/youbin2014/JailbreakDB/resolve/main/text_jailbreak_unique.csv)
- text_regular_unique.csv (https://huggingface.co/datasets/youbin2014/JailbreakDB/resolve/main/text_regular_unique.csv)

### Шаги пайплайна 
Шаги запускаются вручную из терминала: 
- step-1_prepare_data.py - Трансформация данных, подготовка сплитованного набора датасетов для обучения проверок и сохранение их в ./data/processed
- step-2_train_baseline_model.py - Обучение baseline модели TF-IDF + Logreg и сохранение артефактов в ./models/baseline/
- step-3_train_target-base_model.py - Обучение target-base модели ModernBERT и сохранение артефактов в ./models/target-base/ ; target-base - это целевая модель с базовыми параметрами, которые могут быть улучшены на следующем этапе.

### Отчёты
В папку ./reports разово сохранены отчеты подготовки данных и обучения моделей для примера; эта папка отправляется в git.
Основные артефакты по результатам обучения моделей располагаются в папке ./models/ и они не попадают в репозиторий.
На этапе PoC каждому пользователю необходимо самостоятельно запустить пайплайн локально. 
На этапе MVP артефакты будут вынесены в S3. 
