import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import textwrap
import numpy as np
import json
import requests
from requests.auth import HTTPBasicAuth
import html
import psycopg2
import paramiko

#=================================================================================== СОЗДАНИЕ ДИРЕКТОРИЙ

def makeAllDirs(make_data_directory, make_table_rows):
    if not os.path.exists(make_data_directory):
        os.makedirs(make_data_directory)
        
    with open(make_table_rows, 'w', encoding='utf-8') as file:
        file.write('') 
    
make_data_directory = 'data'
make_table_rows = 'table_rows.txt'
    
makeAllDirs(make_data_directory, make_table_rows)
print("Все необходимые директории созданы")

#=================================================================================== ПРОХОД ПО СТРАНИЦАМ

def fetch_and_save(api_url, file_path, auth):
    
    
    users = []
    page = 1

    while page:
        response = requests.get(f"{api_url}?page={page}", auth=auth)
        if response.status_code == 200:
            data = response.json()
            if data and 'results' in data and data['results']:
                users.extend(data['results'])
                page += 1
            
            elif(response.status_code == 404):
                break
        else:
            print(f"Ошибка при получении данных из {api_url}: ", response.status_code)
            break
        

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=4)

    print(f"Данные о {len(users)} пользователях сохранены в файл: {file_path}")

#=================================================================================== ПОДКЛЮЧЕНИЕ К ШИНЕ ДАННЫХ

username = '***'
password = '******'
auth = HTTPBasicAuth(username, password)

api_url1 = "https://esb.fa.ru/api/zkguperson/"
zkguperson_file_path = "./data/zkguperson.json"
fetch_and_save(api_url1, zkguperson_file_path, auth)

api_url2 = "https://esb.fa.ru/api/zkguprofile/"
zkguprofile_file_path = "./data/zkguprofile.json"
fetch_and_save(api_url2, zkguprofile_file_path, auth)

api_url3 = "https://esb.fa.ru/api/zkgudep/"
file_path3 = "./data/zkgudep.json"
fetch_and_save(api_url3, file_path3, auth)

#=================================================================================== ОБЪЕДИНЕНИЕ ПО КЛЮЧАМ

def join_multiple_data(person_file, profile_file, dep_file, final_file):
    with open(person_file, 'r', encoding='utf-8') as f:
        persons_data = json.load(f)

    with open(profile_file, 'r', encoding='utf-8') as f:
        profiles_data = json.load(f)
    
    profiles_dict = {}
    for profile in profiles_data:
        person_id = profile['ID_REC']
        if person_id not in profiles_dict:
            profiles_dict[person_id] = profile

    joined_data = []
    for person in persons_data:
        person_id = person['ID_REC']
        profile = profiles_dict.get(person_id, None)
        if profile:
            person_profile_combined = {**person, **profile}
            joined_data.append(person_profile_combined)

    with open(dep_file, 'r', encoding='utf-8') as f:
        dep_data = json.load(f)
    
    dep_dict = {dep['ID_REC']: dep for dep in dep_data}

    for person in joined_data:
        dep_info = dep_dict.get(person.get('ID_DEP'), None)
        if dep_info:
            person.update(dep_info)

    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(joined_data, f, ensure_ascii=False, indent=4)

    print(f"Итоговые объединенные данные сохранены в файл: {final_file}")

person_file_path = "./data/zkguperson.json"
profile_file_path = "./data/zkguprofile.json"
dep_file_path = "./data/zkgudep.json"
final_file_path = "./data/joined.json"

join_multiple_data(person_file_path, profile_file_path, dep_file_path, final_file_path)


#=================================================================================== УДАЛЕНИЕ НЕОБЯЗАТЕЛЬНЫХ ДАННЫХ

def clean_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data_filtered = [entry for entry in data if 'TITLE' in entry and entry['TITLE'] != "Ректорат"]

    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(data_filtered, file, ensure_ascii=False, indent=4)


json_file_path = './data/joined.json'
clean_data(json_file_path)

#=================================================================================== ФОРМИРОВАНИЕ СПИСКА ФАКУЛЬТЕТОВ И ВХОДЯЩИХ В НИХ КАФЕДР

facult_data = {
    "Факультет международных экономических отношений": {
        "кафедры": [
            "Кафедра международного бизнеса Факультета международных экономических отношений",
            "Кафедра мировой экономики и мировых финансов Факультета международных экономических отношений",
            "Кафедра иностранных языков и межкультурной коммуникации Факультета международных экономических отношений"
        ]
    },
    "Финансовый факультет": {
        "кафедры": [
            "Кафедра банковского дела и монетарного регулирования Финансового факультета",
            "Кафедра общественных финансов Финансового факультета",
            "Кафедра страхования и экономики социальной сферы Финансового факультета",
            "Кафедра финансовых рынков и финансового инжиниринга Финансового факультета",
            "Кафедра 'Финансовые технологии' Финансового факультета",
            "Кафедра 'Финансовый контроль и казначейское дело' Финансового факультета",
            "Базовая кафедра 'ВТБ: современная практика и технологии банковского бизнеса' Финансового факультета",
            "Базовая кафедра 'Ингосстрах' Финансового факультета",
            "Базовая кафедра 'Ипотечное жилищное кредитование и финансовые инструменты рынка недвижимости' Финансового факультета",
            "Базовая кафедра 'Счетная палата Российской Федерации. Государственный аудит' Финансового факультета"
        ]
    },
    "Факультет экономики и бизнеса": {
        "кафедры": [
            "Кафедра корпоративных финансов и корпоративного управления Факультета экономики и бизнеса",
            "Кафедра экономической безопасности и управления рисками Факультета экономики и бизнеса",
            "Кафедра отраслевых рынков Факультета экономики и бизнеса",
            "Кафедра туризма и гостиничного бизнеса Факультета экономики и бизнеса",
            "Кафедра логистики Факультета экономики и бизнеса",
            "Базовая кафедра 'Экономика интеллектуальной собственности' Факультета экономики и бизнеса",
            "Бизнес-школа Финуниверситета (Институт) Факультета экономики и бизнеса",
            "Высшая школа предпринимательства Факультета экономики и бизнеса Факультета экономики и бизнеса"
        ]
    },
    "Факультет 'Высшая школа управления'": {
        "кафедры": [
            'Кафедра менеджмента Факультета "Высшая школа управления',
            'Кафедра стратегического и инновационного развития Факультета "Высшая школа управления',
            'Кафедра маркетинга и спортивного бизнеса Факультета "Высшая школа управления',
            'Кафедра финансового и инвестиционного менеджмента Факультета "Высшая школа управления',
            'Кафедра "Государственное и муниципальное управление" Факультета "Высшая школа управления',
            'Базовая кафедра "Государственно-частное партнерство" Факультета "Высшая школа управления',
            'Базовая кафедра "ПСБ" Факультета "Высшая школа управления',
            'Центр отраслевых исследований и консалтинга Факультета "Высшая школа управления'
        ]
    },
    "Факультет информационных технологий и анализа больших данных": {
        "кафедры": [
            'Кафедра анализа данных и машинного обучения Факультета информационных технологий и анализа больших данных',
            'Кафедра бизнес-информатики Факультета информационных технологий и анализа больших данных',
            'Кафедра информационной безопасности Факультета информационных технологий и анализа больших данных',
            'Кафедра математики Факультета информационных технологий и анализа больших данных',
            'Базовая кафедра "Технологии 1С" Факультета информационных технологий и анализа больших данных',
            'Базовая кафедра "Банковская автоматизация и информационные технологии" Факультета информационных технологий и анализа больших данных',
            'Базовая кафедра Альфа-Банка​ Факультета информационных технологий и анализа больших данных',
            'Кафедра "Системный анализ в экономике Факультета информационных технологий и анализа больших данных"'
        ]
    },
    "Факультет налогов, аудита и бизнес-анализа": {
        "кафедры": [
            "Кафедра аудита и корпоративной отчетности Факультета налогов, аудита и бизнес-анализа",
            "Кафедра бизнес-аналитики Факультета налогов, аудита и бизнес-анализа",
            "Кафедра налогов и налогового администрирования Факультета налогов, аудита и бизнес-анализа",
            'Базовая кафедра "Технологии Доверия" Факультета налогов, аудита и бизнес-анализа',
            'Базовая кафедра "Кэпт "',
            'Базовая кафедра "Группа компаний Б1" Факультета налогов, аудита и бизнес-анализа'
        ]
    },
    "Факультет социальных наук и массовых коммуникаций": {
        "кафедры": [
            "Кафедра гуманитарных наук Факультета социальных наук и массовых коммуникаций",
            "Кафедра социологии Факультета социальных наук и массовых коммуникаций",
            "Кафедра политологии Факультета социальных наук и массовых коммуникаций",
            "Кафедра массовых коммуникаций и медиабизнеса Факультета социальных наук и массовых коммуникаций",
            "Кафедра психологии и развития человеческого капитала Факультета социальных наук и массовых коммуникаций"
        ]
    },
    "Юридический факультет": {
        "кафедры": [
            "Кафедра правового регулирования экономической деятельности Юридического факультета",
            "Кафедра международного и публичного права Юридического факультета",
            'Базовая кафедра "Федеральная антимонопольная служба" Юридического факультета'
        ]
    },
    "Подготовительный факультет": {
        "кафедры": []
    },
    "Институт открытого образования": {
        "кафедры": []
    }
}

#=================================================================================== ДОБАВЛЕНИЕ ФАКУЛЬТЕТОВ

def convert_json_to_facultets(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    facultets = [[entry['EMAIL'].lower().split('@')[0], entry['TITLE']] for entry in data]
    return facultets

#===================================================================================

def fetch_and_parse(url):
    response = requests.get(url)
    if response.status_code == 200:
        return html.unescape(response.text)
    else:
        print(f"Ошибка при доступе к странице: {response.status_code}")
        return None

json_file_path = './data/joined.json'
facultets = convert_json_to_facultets(json_file_path)
cafedra_dict = {name: cafedras for name, *cafedras in facultets}


#=================================================================================== ПОДКЛЮЧЕНИЕ К БАЗЕ ДАННЫХ

db_params = {
    'dbname': '***',
    'user': '*******',
    'password': '********',
    'host': 'localhost',
    'port': ''
}

def fetch_user_stats():
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute('SELECT id, LOWER(login), route, counter FROM user_stats')
        records = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return records
    except Exception as e:
        print(f"Ошибка: {e}")
        return None

#=================================================================================== ПРЕОБРАЗОВАНИЕ ДАННЫХ И ОТРИСОВКА ГРАФИКОВ

def process_and_save_data(data, file_path, cafedra_dict):
    user_data = {}
    for entry in data:
        name = entry[1].replace('@fa.ru', '').lower()
        
        cafedras_info = cafedra_dict.get(name, ["Нет информации"])
        cafedras_str = ", ".join(cafedras_info)
        
        task = entry[2]
        count = entry[3]
        
        task_info = f"{task}: {count}"
        
        if name not in user_data:
            user_data[name] = [f"Кафедры: {cafedras_str}"]
        user_data[name].append(task_info)

    result = ""
    for name, tasks in user_data.items():
        result += f"{name}\n" + "\n".join(tasks) + "\n\n"

    with open(file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(result)

    print(f"Данные успешно сохранены в файл {file_path}.")

json_file_path = './data/joined.json'
facultets = convert_json_to_facultets(json_file_path)
cafedra_dict = {name: cafedras for name, *cafedras in facultets}

data = fetch_user_stats()
output_file_path = './table_rows.txt'

if data:
    process_and_save_data(data, output_file_path, cafedra_dict)
    
file_path = './table_rows.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

data_dict = {}
current_name = ''
for line in lines:
    if line.strip() and not line.startswith('Кафедры:'):
        if ':' in line:
            task, count = line.strip().split(':')
            count = int(count)
            if count > 0:
                if current_name not in data_dict:
                    data_dict[current_name] = {}
                data_dict[current_name][task] = count
        else:
            current_name = line.strip()

df = pd.DataFrame.from_dict(data_dict, orient='index')
    
teachers_directory = 'teachers'
if not os.path.exists(teachers_directory):
    os.makedirs(teachers_directory)

base_color = np.array([76, 114, 176]) / 255

for teacher, data in df.iterrows():
    data = data[data > 0]
    if data.empty:
        continue
    
    shades = [base_color * (0.9 + i/(2*len(data))) for i in range(len(data))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data.plot(kind='barh', ax=ax, color=shades)
    ax.set_title(teacher)
    ax.grid(axis='x', linestyle='--', color='gray', which='both')
    
    plt.tight_layout()
    
    chart_image_path = os.path.join(teachers_directory, f'{teacher}.png')
    plt.savefig(chart_image_path)
    plt.close()

print(f"Графики успешно сохранены в папке '{teachers_directory}'.")
    
file_path = './table_rows.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

cafedra_actions = defaultdict(lambda: defaultdict(int))

for line in lines:
    line = line.strip()
    if line.startswith('Кафедры:'):
        current_cafedras = line.split(':')[1].split(',')
        current_cafedras = [cafedra.strip() for cafedra in current_cafedras if cafedra.strip() and cafedra != "Нет информации"]
    elif ':' in line and current_cafedras:
        action, count = line.split(':')
        action = action.strip()
        count = int(count.strip())
        for cafedra in current_cafedras:
            cafedra_actions[cafedra][action] += count

for cafedra, actions in cafedra_actions.items():
    if cafedra == "Нет информации" or cafedra == "Информация о кафедрах" or cafedra == "Ректорат":
        continue
    df = pd.DataFrame(list(actions.items()), columns=['Action', 'Count']).set_index('Action')
    df = df.sort_values(by='Count', ascending=True)

    plt.figure(figsize=(10, len(actions) * 0.5))
    df['Count'].plot(kind='barh', color='skyblue')
    plt.title(f'Действия на кафедре: {cafedra}')
    plt.xlabel('Количество')
    plt.ylabel('Действия')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.close()
    
images_directory = './cafedras'
if not os.path.exists(images_directory):
    os.makedirs(images_directory)

for cafedra_name, actions in cafedra_actions.items():
    if cafedra_name == "Нет информации" or cafedra_name == "аудита и бизнес-анализа":
        continue

    df = pd.DataFrame([(action, count) for action, count in actions.items() if count > 0],
                      columns=['Action', 'Count'])

    if not df.empty:
        df['Action'] = df['Action'].apply(lambda x: textwrap.fill(x, width=20))

    base_color = np.array([76, 114, 176]) / 255

    shades = [base_color * (0.9 + i/(2*len(df))) for i in range(len(df))]

    plt.figure(figsize=(10, 6))
    plt.barh(df['Action'], df['Count'], color=shades)
    plt.title(textwrap.fill(cafedra_name, width=70))

    plt.grid(axis='x', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    chart_image_path = os.path.join(images_directory, f'{cafedra_name}.png')
    plt.savefig(chart_image_path)
    plt.close()

total_directory = './total'
if not os.path.exists(total_directory):
    os.makedirs(total_directory)

total_actions = {}
for actions in cafedra_actions.values():
    for action, count in actions.items():
        if action != "Нет информации":
            total_actions[action] = total_actions.get(action, 0) + count

df_total = pd.DataFrame(list(total_actions.items()), columns=['Action', 'Count'])
df_total = df_total[df_total['Count'] > 0]
df_total['Action'] = df_total['Action'].apply(lambda x: textwrap.fill(x, width=20))

df_total.sort_values('Count', ascending=False, inplace=True)

base_color = np.array([76, 114, 176]) / 255

shades = [base_color * (0.9 + i/(2*len(df_total))) for i in range(len(df_total))]

plt.figure(figsize=(10, 6))
plt.barh(df_total['Action'], df_total['Count'], color=shades)
plt.title('Общая активность всех кафедр')
plt.grid(axis='x', linestyle='--', linewidth=0.5)
plt.tight_layout()

total_chart_path = os.path.join(total_directory, 'total_cafedra_activity.png')
plt.savefig(total_chart_path)
plt.close()

#=================================================================================== ОБЪЕДИНЕНИЕ ДАННЫХ ПО КАФЕДРАМ

def merge_cafedra_data(table_rows_path, facult_data, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with open(table_rows_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    cafedra_actions = defaultdict(lambda: defaultdict(int))
    faculty_actions = defaultdict(lambda: defaultdict(int))

    cafedra_to_faculty = {}
    for faculty, data in facult_data.items():
        for cafedra in data['кафедры']:
            cafedra_to_faculty[cafedra] = faculty

    current_cafedras = []
    for line in lines:
        line = line.strip()
        if line.startswith('Кафедры:'):
            current_cafedras = line.split(':')[1].split(',')
            current_cafedras = [cafedra.strip() for cafedra in current_cafedras if cafedra.strip() and cafedra != "Нет информации"]
        elif ':' in line and current_cafedras:
            action, count = line.split(':')
            action = action.strip()
            count = int(count.strip())
            for cafedra in current_cafedras:
                faculty = cafedra_to_faculty.get(cafedra)
                if faculty:
                    faculty_actions[faculty][action] += count

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(dict(faculty_actions), file, ensure_ascii=False, indent=4)

    for faculty, actions in faculty_actions.items():
        actions = {action: count for action, count in actions.items() if count > 0}
        actions = dict(sorted(actions.items(), key=lambda item: item[1], reverse=True))
        
        plt.figure(figsize=(10, len(actions)))
        plt.barh(list(actions.keys()), list(actions.values()))
        plt.title(faculty)
        plt.grid(axis='x')
        plt.tight_layout()
        
        faculties_directory = './faculties'
        if not os.path.exists(faculties_directory):
            os.makedirs(faculties_directory)
        plt.savefig(os.path.join(faculties_directory, f'{faculty}.png'))
        plt.close()

    print(f"Объединенные данные сохранены в файл: {output_file_path}")

table_rows_path = './table_rows.txt'
output_file_path = './data/merged_faculty_actions.json'

merge_cafedra_data(table_rows_path, facult_data, output_file_path)

#=================================================================================== ПОСТРОЕНИЕ ОБЩИХ ГРАФИКОВ ПО ВСЕМ ФАКУЛЬТЕТАМ

def calculate_total_faculty_activities(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        faculty_actions = json.load(file)

    total_activities = defaultdict(int)
    for faculty, actions in faculty_actions.items():
        for action, count in actions.items():
            total_activities[faculty] += count

    sorted_total_activities = dict(sorted(total_activities.items(), key=lambda item: item[1], reverse=True))

    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(sorted_total_activities, file, ensure_ascii=False, indent=4)

    print(f"Общее количество действий по факультетам сохранено в файл: {output_file_path}")

input_file_path = './data/merged_faculty_actions.json'
output_file_path = './data/total_faculty_activities.json'
calculate_total_faculty_activities(input_file_path, output_file_path)

#=================================================================================== СОХРАНЕНИЕ ГРАФИКОВ ПО ВСЕМ ФАКУЛЬТЕТАМ

def plot_total_faculty_activities(input_file_path, output_directory):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        total_activities = json.load(file)

    base_color = np.array([76, 114, 176]) / 255
    shades = [base_color * (0.9 + i / (2 * len(total_activities))) for i in range(len(total_activities))]

    faculties = list(total_activities.keys())
    activities_count = list(total_activities.values())

    plt.figure(figsize=(10, 6))
    plt.barh(faculties, activities_count, color=shades)
    plt.xlabel('Количество действий')
    plt.title('Общее количество действий по факультетам')
    plt.grid(axis='x', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file_name = 'total_acts.png'
    output_file_path = os.path.join(output_directory, output_file_name)
    plt.savefig(output_file_path)
    plt.close()

    print(f"График всех действий по факультетам сохранен в папке: {output_directory}, под названием: {output_file_name}")

input_file_path = './data/total_faculty_activities.json'
output_directory = './total'
plot_total_faculty_activities(input_file_path, output_directory)

#=================================================================================== ПОСТРОЕНИЕ ОБЩИХ ГРАФИКОВ ПО ВСЕМ ПОЛЬЗОВАТЕЛЯМ

def plot_faculty_users(input_file_path, output_directory):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        faculty_actions = json.load(file)

    faculty_users = {faculty: len(actions) for faculty, actions in faculty_actions.items()}

    sorted_faculty_users = dict(sorted(faculty_users.items(), key=lambda item: item[1], reverse=True))

    faculties = list(sorted_faculty_users.keys())
    users_count = list(sorted_faculty_users.values())

    base_color = np.array([76, 114, 176]) / 255
    shades = [base_color * (0.9 + i/(2*len(users_count))) for i in range(len(users_count))]

    plt.figure(figsize=(10, len(faculties) / 2))
    plt.barh(faculties, users_count, color=shades)
    plt.xlabel('Количество пользователей')
    plt.title('Общее количество пользователей по факультетам')
    plt.tight_layout()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file_path = os.path.join(output_directory, 'faculty_users.png')
    plt.savefig(output_file_path)
    plt.close()

    print(f"График сохранен в папке: {output_directory}")

input_file_path = './data/merged_faculty_actions.json'
output_directory = './total'
plot_faculty_users(input_file_path, output_directory)

#=================================================================================== СРЕДНЕЕ КОЛИЧЕСТВО ДЕЙСТВИЙ НА ПОЛЬЗОВАТЕЛЯ

def calculate_average_user_activity(input_file_path, output_directory, facult_data):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        faculty_actions = json.load(file)

    faculty_users_count = {faculty: len(data['кафедры']) for faculty, data in facult_data.items()}

    average_activities = {}
    for faculty, actions in faculty_actions.items():
        if isinstance(actions, dict):
            total_actions = sum(actions.values())
        else:
            total_actions = actions

        users_count = faculty_users_count.get(faculty, 1)
        average_activities[faculty] = total_actions / users_count

    sorted_average_activities = dict(sorted(average_activities.items(), key=lambda item: item[1], reverse=True))

    faculties = list(sorted_average_activities.keys())
    average_count = list(sorted_average_activities.values())

    base_color = np.array([76, 114, 176]) / 255
    shades = [base_color * (0.9 + i/(2*len(average_count))) for i in range(len(average_count))]

    plt.figure(figsize=(13, len(faculties) / 2))
    plt.barh(faculties, average_count, color=shades)
    plt.xlabel('Среднее количество действий на пользователя')
    plt.title('Среднее количество действий на пользователя по факультетам')
    plt.tight_layout()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_file_path = os.path.join(output_directory, 'user_activity_avg.png')
    plt.savefig(output_file_path)
    plt.close()

    print(f"График среднего количества действий на пользователя сохранен в папке: {output_directory}")

input_file_path = './data/total_faculty_activities.json'
output_directory = './total'
calculate_average_user_activity(input_file_path, output_directory, facult_data)

#===================================================================================