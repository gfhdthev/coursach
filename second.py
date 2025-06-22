import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ОБРАБОТКА ПУСТЫХ СТРОК
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# Загрузка данных из CSV-файла с указанием разделителя ';'
# Файл содержит маркетинговые данные банка с различными характеристиками клиентов
df = pd.read_csv('bank-additional-full.csv', delimiter=';')

# Проверяем исходное количество строк в датафрейме
initial_row_count = len(df)
print("="*50)
print("РЕЗУЛЬТАТ УДАЛЕНИЯ ПОЛНОСТЬЮ ПУСТЫХ СТРОК")
print("="*50)
print(f"Исходное количество строк: {initial_row_count}")

# Удаляем полностью пустые строки (где все значения NaN)
# Параметр how='all' указывает удалять только строки, где ВСЕ значения отсутствуют
df = df.dropna(how='all')

# Проверяем количество строк после удаления пустых
cleaned_row_count = len(df)
print(f"Количество строк после удаления пустых: {cleaned_row_count}")
print(f"Удалено строк: {initial_row_count - cleaned_row_count}")


'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ОБРАБОТКА СТРОК <10%
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# Сохранение исходной статистики для вывода
initial_rows = len(df)

# Замена специальных значений на NaN для корректной обработки пропусков
# 'unknown' и 'nonexistent' трактуются как отсутствующие данные
df.replace(['unknown', 'nonexistent'], np.nan, inplace=True)

# Расчет порога для удаления строк: удаляем строки с >90% пустых ячеек
# Для этого определяем минимальное количество заполненных ячеек для сохранения строки
n_columns = len(df.columns)  # Общее количество столбцов в датафрейме
min_filled = math.ceil(n_columns * 0.1)  # Минимум 10% заполненных ячеек

# Фильтрация строк: сохраняем только строки с количеством заполненных ячеек >= min_filled
# Используем параметр thresh в dropna() для установки минимального количества непустых значений
df = df.dropna(thresh=min_filled)

# Расчет и вывод статистики обработки
final_rows = len(df)
removed_rows = initial_rows - final_rows
removed_percentage = (removed_rows / initial_rows) * 100

print("="*50)
print("РЕЗУЛЬТАТЫ ПРЕДОБРАБОТКИ ДАННЫХ, ГДЕ МЕНЬШЕ 10% ИНФОРМАЦИИ В СТРОКЕ")
print("="*50)
print(f"Исходное количество строк: {initial_rows}")
print(f"Сохранено строк: {final_rows}")
print(f"Удалено строк: {removed_rows} ({removed_percentage:.2f}% от общего объема)")
print(f"Причина удаления: более 90% пустых ячеек в строке")
print(f"Порог заполненности: минимум {min_filled} заполненных ячеек из {n_columns}")

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ПРОВЕРКА НА СТОЛБЦЫ С ОДНИМ ЗНАЧЕНИЕМ
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# Создание словаря для группировки столбцов по их содержимому
column_groups = {}
duplicates_found = False

# Проверка каждого столбца на уникальность
print("="*50)
print("ПРОВЕРКА НА ИДЕНТИЧНЫЕ СТОЛБЦЫ")
print("="*50)

# Сравнение столбцов попарно
duplicate_pairs = []
for i, col1 in enumerate(df.columns):
    for j, col2 in enumerate(df.columns[i+1:], i+1):
        if df[col1].equals(df[col2]):
            duplicate_pairs.append((col1, col2))
            duplicates_found = True

# Вывод результатов
if duplicates_found:
    print("Обнаружены идентичные столбцы:")
    for pair in duplicate_pairs:
        print(f"- Столбец '{pair[0]}' полностью совпадает со столбцом '{pair[1]}'")
else:
    print("Идентичные столбцы не обнаружены")
    
print(f"Всего столбцов в датасете: {len(df.columns)}")
print(f"Количество идентичных пар столбцов: {len(duplicate_pairs)}")

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ПРОВЕРКА НА СТОЛБЦЫ , ЧТО В НИХ > 10% РАЗНОЙ ИНФОРМАЦИИ
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# Загрузка данных
df = pd.read_csv('bank-additional-full.csv', sep=';', quotechar='"')

# Анализ столбцов с доминирующими значениями
print("="*50)
print("АНАЛИЗ СТОЛБЦОВ С ДОМИНИРУЮЩИМИ ЗНАЧЕНИЯМИ")
print("="*50)

# Список проблемных столбцов для удаления
columns_to_remove = []
threshold = 0.9

for column in df.columns:
    # Рассчет доли наиболее частого значения
    value_counts = df[column].value_counts(normalize=True)
    most_common_ratio = value_counts.iloc[0]
    
    # Проверка порога
    if most_common_ratio > threshold:
        print(f"- Столбец '{column}' (тип: {df[column].dtype}):")
        print(f"  Доминирующее значение: '{value_counts.index[0]}'")
        print(f"  Процент встречаемости: {most_common_ratio*100:.2f}%")
        columns_to_remove.append(column)

# Удаление проблемных столбцов
if columns_to_remove:
    print("\nУдаление столбцов с низкой информационной ценностью:")
    print(f"- Столбцы для удаления: {', '.join(columns_to_remove)}")
    df = df.drop(columns=columns_to_remove)
    print(f"Обновленное количество столбцов: {len(df.columns)}")
else:
    print("\nСтолбцы для удаления не обнаружены")

# Вывод статистики
print(f"\nВсего проверено столбцов: {len(df.columns) + len(columns_to_remove)}")
print(f"Проблемных столбцов: {len(columns_to_remove)}")
print("="*50)

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
УДАЛЕНИЕ 16-20 СТОЛБЦОВ Т.К. ОНИ НЕ НЕСУТ СМЫСЛОВОЙ НАГРУЗКИ
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# Определение столбцов для удаления
columns_to_drop = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Проверка наличия столбцов перед удалением
existing_columns = [col for col in columns_to_drop if col in df.columns]

# Удаление столбцов
if existing_columns:
    print(f"Удаление столбцов: {', '.join(existing_columns)}")
    df = df.drop(columns=existing_columns)
    print(f"Обновленное количество столбцов: {len(df.columns)}")
else:
    print("Указанные столбцы не найдены в датафрейме")

# Проверка результата
print("\nПервые 3 строки после удаления:")
print(df.head(3))

df.to_csv('result_after_2_chapter.csv')

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
3 ГЛАВА
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
КАКОЙ ПРОЦЕНТ ЛЮДЕЙ ВЗЯЛИ ДЕПОЗИТ
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# Расчет распределения
deposit_distribution = df['y'].value_counts(normalize=True) * 100

# Построение круговой диаграммы
plt.figure(figsize=(10, 7))
colors = ['#ff9999', '#66b3ff']
explode = (0.1, 0)  # Выделение сегмента "yes"

patches, texts, autotexts = plt.pie(
    deposit_distribution,
    labels=['No deposit', 'Deposit'],
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    explode=explode,
    shadow=True
)

# Настройка стиля
plt.setp(autotexts, size=12, weight='bold')
plt.setp(texts, size=14)
plt.title('Распределение подписки на депозит', fontsize=16, pad=20)
plt.axis('equal')

# Сохранение и отображение
plt.tight_layout()
plt.savefig('deposit_distribution.png', dpi=300)
plt.show()

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ЗАВИСИМОСТЬ ДЕПОЗИТА ОТ ВОЗРАСТА
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# Фильтруем данные
all_clients = df['age']
deposit_clients = df[df['y'] == 'yes']['age']

# Создаем гистограмму
plt.figure(figsize=(14, 8))
bins = np.arange(15, 70, 3)

# Общее распределение клиентов
plt.hist(all_clients, bins=bins, alpha=0.7, color='skyblue', 
        edgecolor='black', label='Все клиенты')

# Распределение клиентов с депозитом
plt.hist(deposit_clients, bins=bins, alpha=0.9, color='coral', 
        edgecolor='black', label='Клиенты с депозитом')

# Настройка оформления
plt.title('Распределение клиентов по возрасту', fontsize=16, pad=20)
plt.xlabel('Возраст (лет)', fontsize=12)
plt.ylabel('Количество клиентов', fontsize=12)
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig('age_distribution.png', dpi=300)
plt.show()

'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
ЗАВИСИМОСТЬ ДЕПОЗИТА ОТ образования
!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# Фильтруем данные: исключаем образование "unknown"
df_filtered = df[df['education'] != 'unknown']

# Создаем порядок категорий образования
education_order = [
    'illiterate',
    'basic.4y',
    'basic.6y',
    'basic.9y',
    'high.school',
    'professional.course',
    'university.degree'
]

# Подсчет клиентов по образованию
education_counts = df_filtered['education'].value_counts().reindex(education_order).fillna(0)
deposit_counts = df_filtered[df_filtered['y'] == 'yes']['education'].value_counts().reindex(education_order).fillna(0)

# Создаем график
plt.figure(figsize=(14, 8))

# Позиции для столбцов
x = np.arange(len(education_order))
width = 0.35

# Столбцы для всех клиентов и клиентов с депозитом
plt.bar(x - width/2, education_counts, width, 
        color='skyblue', edgecolor='black', label='Все клиенты')
plt.bar(x + width/2, deposit_counts, width, 
        color='coral', edgecolor='black', label='Клиенты с депозитом')

# Подписи и оформление
plt.title('Распределение клиентов по уровню образования (исключены клиенты с неизвестным образованием)', fontsize=16, pad=20)
plt.xlabel('Уровень образования', fontsize=12)
plt.ylabel('Количество клиентов', fontsize=12)
plt.xticks(x, [
    'Неграмотные', 'Базовое (4 года)', 'Базовое (6 лет)', 
    'Базовое (9 лет)', 'Средняя школа', 'Проф. курсы', 
    'Университет'
], rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Добавляем проценты
total_clients = df_filtered.shape[0]
for i in range(len(education_counts)):
    count = education_counts.iloc[i]
    percentage = (count / total_clients) * 100
    plt.text(x[i] - width/2, count + 20, f'{percentage:.1f}%', 
            ha='center', fontsize=9)

for i in range(len(deposit_counts)):
    count = deposit_counts.iloc[i]
    group_count = education_counts.iloc[i]
    if count > 0 and group_count > 0:
        percentage = (count / group_count) * 100
        plt.text(x[i] + width/2, count + 10, f'{percentage:.1f}%', 
                ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('education_distribution_filtered.png', dpi=300)
plt.show()