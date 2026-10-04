import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#загружаем файл ксв
file_path=''
df = pd.read_csv(url)

print("1.Информация о датасете.")
df.info()
print("\nКол-во пропусков в признаках:")
print (df.isnull().sum())

#создаём диаграмму:здоровых и больных
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target', palette='Set2')
plt.title('Распределение больных и здоровых пациентов')
plt.xlabel('Статус (0 = Здоров, 1 = Болен)')
plt.ylabel('Количество пациентов')
plt.xticks([0,1], ['Здоровые', 'Больные'])
plt.show()

#строим диаграмму рассеяния: thalach от age с расцветкой по таргет
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x= 'age', y= 'thalach', hue= 'target', palette='coolwarm', alpha=0.8)
plt.title('Зависимость максимального пульса от возраста')
plt.xlabel('Возраст(age)')
plt.ylabel('Максимальный пульс(thalach)')
plt.show()

#преобразовываем признаки sex и One-Hot Encoding
df['sex'] = df['sex'].map({0:'female', 1:'male'})
df= pd.get_dummies(df, columns=['sex'], drop_first=False)
print("\n--Результат One-Hot Encoding(первые 5 строк)--")
print(df[['sex_female', 'sex_male']].head())

#средний уровень холестерина(chol) для больных и здоровых
mean_chol = df.groupby('target'[chol].mean())
print("\n--Средний уровень холестерина(chol)--")
print(f"Здоровые пациенты(target 0): {mean_chol[0]:.2f}")
print(f"Больные пациенты(target 1): {mean_chol[1]:.2f}")

#нормализация признаков age, trestbps, chol, thalach
scaler = MinMaxScaler()
features_to_scale = ['age', 'trestbps', 'chol', 'thalach']
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

print("\n--Результат нормализации(первые 5 строк)--")
print(df[features_to_scale].head())