import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from faker import Faker

np.random.seed(42)
fake = Faker()

n = 5000

names = [fake.name() for _ in range(n)]
phone_numbers = [fake.phone_number() for _ in range(n)]
emails = [fake.email() for _ in range(n)]
addresses = [fake.address().replace('\n', ', ') for _ in range(n)]
birth_dates = [fake.date_of_birth(minimum_age=18, maximum_age=80) for _ in range(n)]
salaries = np.random.randint(30000, 150000, n)
departments = np.random.choice(['HR', 'IT', 'Finance', 'Marketing', 'Sales', 'Operations'], n)
genders = np.random.choice(['Male', 'Female', 'Non-binary'], n)

df = pd.DataFrame({
    'Name': names,
    'Phone Number': phone_numbers,
    'Email': emails,
    'Address': addresses,
    'Birth Date': birth_dates,
    'Salary': salaries,
    'Department': departments,
    'Gender': genders
})

df['Phone Number Cleaned'] = df['Phone Number'].apply(lambda x: re.sub(r'\D', '', x))
df['Email Domain'] = df['Email'].apply(lambda x: re.search(r'@(\w+\.\w+)', x).group(1))
df['Age'] = df['Birth Date'].apply(lambda x: 2024 - x.year)
df['State'] = df['Address'].apply(lambda x: re.search(r', ([A-Z]{2}) \d{5}', x).group(1) if re.search(r', ([A-Z]{2}) \d{5}', x) else 'Unknown')

plt.figure(figsize=(14, 7))
sns.histplot(df['Age'], bins=30, kde=True, color='blue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(14, 7))
sns.boxplot(x='Department', y='Salary', data=df)
plt.title('Salary Distribution by Department')
plt.xlabel('Department')
plt.ylabel('Salary')
plt.show()

df['Salary Bracket'] = pd.cut(df['Salary'], bins=[0, 50000, 100000, 150000], labels=['Low', 'Medium', 'High'])
dept_gender_salary = df.pivot_table(index='Department', columns='Gender', values='Salary', aggfunc=np.mean)

plt.figure(figsize=(14, 7))
sns.heatmap(dept_gender_salary, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title('Average Salary by Department and Gender')
plt.show()

df['Phone Valid'] = df['Phone Number Cleaned'].apply(lambda x: len(x) == 10)
df['Email Valid'] = df['Email'].apply(lambda x: bool(re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', x)))

plt.figure(figsize=(14, 7))
sns.countplot(x='Gender', hue='Phone Valid', data=df)
plt.title('Phone Number Validity by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(14, 7))
sns.countplot(x='Gender', hue='Email Valid', data=df)
plt.title('Email Validity by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(14, 7))
sns.countplot(y='State', data=df, order=df['State'].value_counts().index)
plt.title('Count of Entries by State')
plt.xlabel('Count')
plt.ylabel('State')
plt.show()

dept_salary = df.groupby('Department')['Salary'].agg(['mean', 'median', 'std'])
print(dept_salary)

phone_validity = df['Phone Valid'].mean()
email_validity = df['Email Valid'].mean()

print(f"Percentage of Valid Phone Numbers: {phone_validity * 100:.2f}%")
print(f"Percentage of Valid Emails: {email_validity * 100:.2f}%")

df['Initial'] = df['Name'].apply(lambda x: x.split()[0][0])
initial_gender_count = df.pivot_table(index='Initial', columns='Gender', aggfunc='size', fill_value=0)

plt.figure(figsize=(14, 7))
sns.heatmap(initial_gender_count, annot=True, fmt="d", cmap="coolwarm")
plt.title('Gender Distribution by Name Initial')
plt.show()

df['Salary Category'] = df['Salary'].apply(lambda x: 'High' if x > 100000 else ('Medium' if x > 50000 else 'Low'))
gender_salary_category = df.groupby(['Gender', 'Salary Category']).size().unstack()

plt.figure(figsize=(14, 7))
gender_salary_category.plot(kind='bar', stacked=True, colormap='viridis', width=0.8)
plt.title('Salary Category Distribution by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
