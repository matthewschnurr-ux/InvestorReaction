#!/usr/bin/env python3
"""
Canadian Investor Persona Generator
Generates 1000 demographically proportional synthetic personas of Canadians
looking to invest. Uses Canadian Census and StatsCan data for distributions.
"""

import random
import json
import csv
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

SEED = 42
NUM_PERSONAS = 1000

# ============================================================
# DEMOGRAPHIC DISTRIBUTIONS (based on 2021 Canadian Census / StatsCan)
# ============================================================

# Province population shares (%)
PROVINCE_WEIGHTS = {
    "Ontario": 38.5,
    "Quebec": 23.0,
    "British Columbia": 13.5,
    "Alberta": 11.6,
    "Manitoba": 3.6,
    "Saskatchewan": 3.1,
    "Nova Scotia": 2.6,
    "New Brunswick": 2.1,
    "Newfoundland and Labrador": 1.4,
    "Prince Edward Island": 0.4,
    "Northwest Territories": 0.12,
    "Yukon": 0.12,
    "Nunavut": 0.10,
}

# Major cities/towns by province with relative weights
CITIES_BY_PROVINCE = {
    "Ontario": [
        ("Toronto", 40), ("Ottawa", 10), ("Mississauga", 6), ("Brampton", 5),
        ("Hamilton", 5), ("London", 4), ("Markham", 3), ("Kitchener", 3),
        ("Windsor", 2), ("Oshawa", 2), ("Barrie", 2), ("St. Catharines", 2),
        ("Guelph", 2), ("Cambridge", 1), ("Thunder Bay", 1), ("Sudbury", 1),
        ("Kingston", 1), ("Peterborough", 1), ("Oakville", 2), ("Burlington", 2),
        ("Richmond Hill", 2), ("Vaughan", 2),
    ],
    "Quebec": [
        ("Montreal", 45), ("Quebec City", 15), ("Laval", 8), ("Gatineau", 6),
        ("Longueuil", 5), ("Sherbrooke", 4), ("Saguenay", 3), ("Lévis", 2),
        ("Trois-Rivières", 3), ("Terrebonne", 2), ("Drummondville", 2),
        ("Saint-Jean-sur-Richelieu", 2), ("Rimouski", 1), ("Chicoutimi", 1),
        ("Granby", 1),
    ],
    "British Columbia": [
        ("Vancouver", 35), ("Surrey", 12), ("Burnaby", 8), ("Richmond", 6),
        ("Kelowna", 5), ("Victoria", 8), ("Abbotsford", 4), ("Coquitlam", 4),
        ("Nanaimo", 3), ("Kamloops", 3), ("Langley", 3), ("Prince George", 2),
        ("Chilliwack", 2), ("North Vancouver", 3), ("New Westminster", 2),
    ],
    "Alberta": [
        ("Calgary", 38), ("Edmonton", 35), ("Red Deer", 5), ("Lethbridge", 4),
        ("St. Albert", 3), ("Medicine Hat", 2), ("Airdrie", 3), ("Spruce Grove", 2),
        ("Grande Prairie", 2), ("Fort McMurray", 3), ("Sherwood Park", 3),
    ],
    "Manitoba": [
        ("Winnipeg", 65), ("Brandon", 8), ("Steinbach", 4), ("Thompson", 3),
        ("Portage la Prairie", 3), ("Selkirk", 3), ("Winkler", 3),
        ("Morden", 2), ("Dauphin", 2), ("The Pas", 2),
    ],
    "Saskatchewan": [
        ("Saskatoon", 35), ("Regina", 35), ("Prince Albert", 6), ("Moose Jaw", 5),
        ("Swift Current", 3), ("Yorkton", 3), ("North Battleford", 3),
        ("Estevan", 2), ("Weyburn", 2), ("Lloydminster", 3),
    ],
    "Nova Scotia": [
        ("Halifax", 55), ("Dartmouth", 10), ("Sydney", 8), ("Truro", 5),
        ("New Glasgow", 4), ("Kentville", 3), ("Amherst", 3),
        ("Bridgewater", 3), ("Yarmouth", 3), ("Antigonish", 3),
    ],
    "New Brunswick": [
        ("Moncton", 25), ("Saint John", 25), ("Fredericton", 20),
        ("Dieppe", 5), ("Miramichi", 5), ("Edmundston", 5),
        ("Bathurst", 5), ("Campbellton", 3), ("Riverview", 4),
        ("Oromocto", 3),
    ],
    "Newfoundland and Labrador": [
        ("St. John's", 50), ("Mount Pearl", 10), ("Corner Brook", 10),
        ("Conception Bay South", 8), ("Paradise", 7), ("Grand Falls-Windsor", 5),
        ("Gander", 4), ("Happy Valley-Goose Bay", 3), ("Labrador City", 3),
    ],
    "Prince Edward Island": [
        ("Charlottetown", 50), ("Summerside", 20), ("Stratford", 10),
        ("Cornwall", 10), ("Montague", 5), ("Souris", 5),
    ],
    "Northwest Territories": [
        ("Yellowknife", 55), ("Hay River", 12), ("Inuvik", 10),
        ("Fort Smith", 8), ("Behchoko", 8), ("Norman Wells", 7),
    ],
    "Yukon": [
        ("Whitehorse", 70), ("Dawson City", 10), ("Watson Lake", 8),
        ("Haines Junction", 6), ("Carmacks", 6),
    ],
    "Nunavut": [
        ("Iqaluit", 35), ("Rankin Inlet", 12), ("Arviat", 10),
        ("Baker Lake", 8), ("Cambridge Bay", 8), ("Pond Inlet", 7),
        ("Igloolik", 7), ("Pangnirtung", 7), ("Cape Dorset", 6),
    ],
}

# Age distribution for adults 18+ (midpoints and weights)
AGE_BRACKETS = [
    ((18, 24), 9),
    ((25, 34), 17),
    ((35, 44), 17),
    ((45, 54), 16),
    ((55, 64), 17),
    ((65, 74), 14),
    ((75, 85), 10),
]

GENDERS = [("Male", 49), ("Female", 50), ("Non-binary", 1)]

# National ethnicity distribution (%)
ETHNICITY_NATIONAL = {
    "European": 68,
    "South Asian": 7.5,
    "Chinese": 5,
    "Black": 4.3,
    "Filipino": 3,
    "Arab": 2.2,
    "Latin American": 1.8,
    "Southeast Asian": 1.2,
    "Indigenous": 5,
    "Korean": 0.7,
    "Japanese": 0.3,
    "West Asian": 1.0,
}

# Provincial ethnicity modifiers (multipliers relative to national average)
ETHNICITY_PROVINCE_MOD = {
    "Ontario": {"South Asian": 1.6, "Chinese": 1.2, "Black": 1.3, "Filipino": 1.1, "Arab": 1.3, "Indigenous": 0.5},
    "Quebec": {"European": 1.05, "Arab": 2.0, "Black": 1.2, "Latin American": 1.4, "South Asian": 0.5, "Chinese": 0.6, "Indigenous": 0.5},
    "British Columbia": {"Chinese": 2.4, "South Asian": 1.8, "Filipino": 1.3, "Korean": 1.5, "Japanese": 2.0, "Southeast Asian": 1.3, "Indigenous": 0.8, "Black": 0.4},
    "Alberta": {"South Asian": 1.2, "Filipino": 1.6, "Chinese": 0.9, "Indigenous": 0.9, "Black": 0.8, "Arab": 0.7},
    "Manitoba": {"Indigenous": 3.2, "Filipino": 2.5, "South Asian": 0.8, "Black": 0.7, "Chinese": 0.5},
    "Saskatchewan": {"Indigenous": 3.5, "South Asian": 0.6, "Chinese": 0.5, "Filipino": 0.8, "Black": 0.4},
    "Nova Scotia": {"Black": 1.2, "Indigenous": 1.2, "Arab": 1.0, "South Asian": 0.4, "Chinese": 0.3},
    "New Brunswick": {"Indigenous": 0.8, "European": 1.1, "South Asian": 0.3, "Chinese": 0.3, "Arab": 0.5},
    "Newfoundland and Labrador": {"European": 1.15, "Indigenous": 0.8, "South Asian": 0.2, "Chinese": 0.2},
    "Prince Edward Island": {"European": 1.1, "South Asian": 0.3, "Chinese": 0.3, "Indigenous": 0.3},
    "Northwest Territories": {"Indigenous": 8.0, "European": 0.6, "South Asian": 0.2},
    "Yukon": {"Indigenous": 4.5, "European": 0.8, "South Asian": 0.2},
    "Nunavut": {"Indigenous": 16.0, "European": 0.15, "South Asian": 0.05},
}

# Education levels and weights (adults 25+)
EDUCATION_LEVELS = [
    ("No certificate/diploma", 8),
    ("High school diploma", 25),
    ("College diploma or trades certificate", 33),
    ("Bachelor's degree", 22),
    ("Master's degree", 9),
    ("Doctorate or professional degree", 3),
]

# Employment status by age bracket modifier
EMPLOYMENT_STATUSES = ["Employed full-time", "Employed part-time", "Self-employed", "Retired", "Unemployed", "Student"]

# Occupation categories
OCCUPATIONS = {
    "Management": 10,
    "Business/Finance/Administration": 15,
    "Sciences/Technology/Engineering": 8,
    "Health": 8,
    "Education/Law/Government": 10,
    "Arts/Culture/Recreation/Sport": 3,
    "Sales and Service": 20,
    "Trades/Transport/Equipment": 12,
    "Natural Resources/Agriculture": 4,
    "Manufacturing/Utilities": 5,
    "Information Technology": 5,
}

# Home ownership rates by age group
HOME_OWNERSHIP_BY_AGE = {
    (18, 24): 0.10, (25, 34): 0.40, (35, 44): 0.60,
    (45, 54): 0.72, (55, 64): 0.76, (65, 74): 0.78, (75, 85): 0.74,
}

# Province home price medians (approx in CAD)
HOME_PRICE_BY_PROVINCE = {
    "Ontario": 780000, "Quebec": 420000, "British Columbia": 900000,
    "Alberta": 430000, "Manitoba": 340000, "Saskatchewan": 310000,
    "Nova Scotia": 380000, "New Brunswick": 280000,
    "Newfoundland and Labrador": 260000, "Prince Edward Island": 350000,
    "Northwest Territories": 380000, "Yukon": 480000, "Nunavut": 420000,
}

# City premium multipliers for major cities
CITY_PREMIUM = {
    "Toronto": 1.4, "Vancouver": 1.5, "Victoria": 1.2, "Ottawa": 1.1,
    "Calgary": 1.05, "Montreal": 1.0, "Halifax": 1.0, "Kelowna": 1.15,
    "Richmond": 1.3, "Markham": 1.2, "Richmond Hill": 1.2, "Oakville": 1.3,
    "Burlington": 1.15, "North Vancouver": 1.4, "Vaughan": 1.15,
}

# ============================================================
# NAME POOLS BY ETHNICITY AND GENDER
# ============================================================

FIRST_NAMES = {
    "European": {
        "Male": ["James", "William", "John", "Robert", "Michael", "David", "Richard", "Thomas",
                  "Daniel", "Matthew", "Andrew", "Christopher", "Mark", "Stephen", "Paul",
                  "Patrick", "Kevin", "Brian", "Sean", "Connor", "Liam", "Noah", "Ethan",
                  "Owen", "Alexander", "Ryan", "Scott", "Craig", "Douglas", "Ian", "Colin",
                  "Derek", "Trevor", "Gordon", "Bruce", "Alan", "Keith", "Neil", "Grant", "Ross"],
        "Female": ["Sarah", "Jennifer", "Jessica", "Emily", "Emma", "Olivia", "Sophia", "Charlotte",
                    "Amelia", "Hannah", "Grace", "Elizabeth", "Catherine", "Margaret", "Laura",
                    "Karen", "Lisa", "Nicole", "Michelle", "Rebecca", "Victoria", "Rachel",
                    "Megan", "Ashley", "Heather", "Donna", "Sandra", "Wendy", "Carol", "Diane",
                    "Susan", "Nancy", "Chloe", "Abigail", "Ella", "Natalie", "Allison", "Claire"],
        "Non-binary": ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Quinn", "Avery", "Riley",
                        "Jamie", "Sam", "Drew", "Robin", "Hayden", "Cameron", "Reese", "Emerson"],
    },
    "French Canadian": {
        "Male": ["Jean", "Pierre", "Jacques", "Michel", "François", "André", "Claude", "Marc",
                  "Louis", "Philippe", "Yves", "Alain", "Guy", "René", "Denis", "Bernard",
                  "Serge", "Luc", "Étienne", "Mathieu", "Gabriel", "Alexandre", "Maxime",
                  "Antoine", "Olivier", "Julien", "Nicolas", "Benoît", "Simon", "Émile"],
        "Female": ["Marie", "Isabelle", "Sophie", "Catherine", "Anne", "Julie", "Nathalie",
                    "Sylvie", "Hélène", "Monique", "Céline", "Françoise", "Chantal", "Josée",
                    "Diane", "Madeleine", "Louise", "Claire", "Émilie", "Aurélie", "Camille",
                    "Léa", "Florence", "Gabrielle", "Valérie", "Geneviève", "Véronique"],
        "Non-binary": ["Dominique", "Claude", "Camille", "Maxime", "Alex", "Sacha", "Lou",
                        "Eden", "Charlie", "Noa"],
    },
    "South Asian": {
        "Male": ["Raj", "Amit", "Sanjay", "Vikram", "Arun", "Pradeep", "Rahul", "Suresh",
                  "Deepak", "Nikhil", "Arjun", "Rohan", "Karan", "Naveen", "Harpreet",
                  "Gurpreet", "Manpreet", "Jaspreet", "Ranjit", "Amarjit", "Anand", "Dev",
                  "Hari", "Mohan", "Ajay", "Vivek", "Ashok", "Siddharth", "Kunal", "Tarun"],
        "Female": ["Priya", "Anita", "Sunita", "Lakshmi", "Deepa", "Neha", "Pooja", "Meera",
                    "Kavita", "Aisha", "Simran", "Harleen", "Navneet", "Jasleen", "Shalini",
                    "Rashmi", "Anjali", "Nisha", "Divya", "Asha", "Geeta", "Indira", "Kamala",
                    "Leela", "Mira", "Pallavi", "Rekha", "Seema", "Swati", "Uma"],
        "Non-binary": ["Kiran", "Jaya", "Noor", "Ari", "Sasha", "Devi"],
    },
    "Chinese": {
        "Male": ["Wei", "Jun", "Lei", "Ming", "Jian", "Hao", "Yong", "Feng", "David", "Kevin",
                  "Jason", "Eric", "Tony", "Andy", "Peter", "Alex", "Justin", "Raymond",
                  "Henry", "Daniel", "Victor", "Steven", "Albert", "Frank", "George",
                  "Richard", "Howard", "Eugene", "Philip", "Dennis"],
        "Female": ["Li", "Fang", "Mei", "Xin", "Ying", "Jing", "Yan", "Hui", "Grace", "Amy",
                    "Jenny", "Lucy", "Cindy", "Michelle", "Christina", "Karen", "Nancy",
                    "Angela", "Linda", "Helen", "Carol", "Vivian", "Jessica", "Diana",
                    "Irene", "Gloria", "Wendy", "Joanne", "Annie", "Emily"],
        "Non-binary": ["Yu", "Lin", "Kai", "Sky", "Jamie", "Alex"],
    },
    "Black": {
        "Male": ["Marcus", "Andre", "Kwame", "Jermaine", "Desmond", "Jamal", "Kofi", "Emmanuel",
                  "Chidi", "Oluwaseun", "Aboubacar", "Moussa", "Ibrahim", "Thierry", "Patrick",
                  "Samuel", "Joseph", "David", "Daniel", "Michael", "Anthony", "Jordan",
                  "Christopher", "Jason", "Darren", "Terrence", "Xavier", "Malik", "Isaiah"],
        "Female": ["Keisha", "Aisha", "Nia", "Fatou", "Aminata", "Ngozi", "Chioma", "Adaeze",
                    "Precious", "Grace", "Joy", "Faith", "Blessing", "Nadège", "Fatoumata",
                    "Mariama", "Amina", "Sandra", "Patricia", "Michelle", "Angela", "Nicole",
                    "Jasmine", "Brittany", "Tanya", "Monique", "Simone", "Vanessa", "Tamara"],
        "Non-binary": ["Jordan", "Sage", "Phoenix", "Amari", "Skyler", "Quinn"],
    },
    "Filipino": {
        "Male": ["Jose", "Juan", "Antonio", "Manuel", "Francisco", "Rafael", "Miguel", "Carlos",
                  "Roberto", "Eduardo", "Angelo", "Marco", "Paolo", "Carlo", "Gabriel",
                  "Christian", "Mark", "John Paul", "Jerome", "Benedict"],
        "Female": ["Maria", "Ana", "Rosa", "Carmen", "Luz", "Grace", "Joy", "Faith", "Hope",
                    "Cherry", "April", "Crystal", "Jasmine", "Angelica", "Michelle", "Maricel",
                    "Rowena", "Lea", "Gemma", "Donna"],
        "Non-binary": ["Angel", "Francis", "Jamie", "Rio", "Pat", "Sam"],
    },
    "Arab": {
        "Male": ["Mohammed", "Ahmed", "Ali", "Hassan", "Omar", "Khalil", "Youssef", "Ibrahim",
                  "Karim", "Tariq", "Samir", "Walid", "Nabil", "Fadi", "Rami", "Sami",
                  "Marwan", "Adel", "Amir", "Bilal"],
        "Female": ["Fatima", "Aisha", "Layla", "Noor", "Hana", "Sara", "Yasmin", "Mariam",
                    "Rania", "Dina", "Salma", "Lina", "Rana", "Nada", "Amira", "Dalal",
                    "Mona", "Nadine", "Zeina", "Lamia"],
        "Non-binary": ["Noor", "Salam", "Farah", "Shams", "Ihsan", "Rihan"],
    },
    "Latin American": {
        "Male": ["Carlos", "Miguel", "Jose", "Luis", "Juan", "Diego", "Alejandro", "Roberto",
                  "Fernando", "Ricardo", "Pablo", "Eduardo", "Andres", "Gabriel", "Sebastian",
                  "Mateo", "Santiago", "Daniel", "Nicolas", "Adrian"],
        "Female": ["Maria", "Isabella", "Valentina", "Sofia", "Camila", "Lucia", "Elena",
                    "Rosa", "Carmen", "Adriana", "Gabriela", "Patricia", "Andrea", "Ana",
                    "Carolina", "Laura", "Daniela", "Alejandra", "Mariana", "Natalia"],
        "Non-binary": ["Alex", "Ari", "Angel", "Cruz", "Sol", "Mar"],
    },
    "Southeast Asian": {
        "Male": ["Nguyen", "Tran", "Minh", "Duc", "Thanh", "Anh", "Kiet", "Binh",
                  "Somchai", "Boun", "Phong", "Long", "Tuan", "Huy", "Duy"],
        "Female": ["Mai", "Linh", "Huong", "Thuy", "Lan", "Ha", "Ngoc", "Anh",
                    "Kim", "Thi", "Phuong", "Oanh", "Tam", "Hoa", "Tuyet"],
        "Non-binary": ["An", "Lam", "Tien", "Vien", "Tam", "Bao"],
    },
    "Indigenous": {
        "Male": ["John", "James", "Robert", "William", "David", "Thomas", "Michael", "Joseph",
                  "Daniel", "Raymond", "George", "Frank", "Edward", "Peter", "Alexander",
                  "Matthew", "Nathan", "Tyler", "Jordan", "Jesse", "Kyle", "Brandon", "Trevor",
                  "Curtis", "Daryl", "Wesley", "Russell", "Elijah", "Hunter", "River"],
        "Female": ["Mary", "Sarah", "Margaret", "Catherine", "Elizabeth", "Jennifer", "Jessica",
                    "Emily", "Nicole", "Amanda", "Crystal", "Brittany", "Ashley", "Stephanie",
                    "Michelle", "Laura", "Samantha", "Angela", "Rebecca", "Tiffany", "Autumn",
                    "Dawn", "Willow", "Raven", "Sierra", "Jade", "Dakota", "Cheyenne"],
        "Non-binary": ["River", "Sky", "Dakota", "Sage", "Cedar", "Aspen", "Phoenix", "Wren"],
    },
    "Korean": {
        "Male": ["Sung", "Hyun", "Jin", "Min", "Joon", "Dong", "Young", "Sang", "David",
                  "Daniel", "James", "Brian", "John", "Kevin", "Steven", "Andrew"],
        "Female": ["Soo", "Hye", "Ji", "Eun", "Min", "Yeon", "Seon", "Hana",
                    "Grace", "Jennifer", "Christina", "Sarah", "Michelle", "Jessica", "Emily"],
        "Non-binary": ["Min", "Yoon", "Haru", "Sky", "Alex", "Jun"],
    },
    "Japanese": {
        "Male": ["Takeshi", "Hiroshi", "Kenji", "Yuki", "Ryo", "Ken", "Taro", "Akira",
                  "David", "Kevin", "Mark", "Steven", "Brian", "Paul", "Ryan"],
        "Female": ["Yuko", "Keiko", "Naomi", "Mika", "Emi", "Sakura", "Ayumi", "Rina",
                    "Lisa", "Karen", "Linda", "Susan", "Jennifer", "Emily", "Sarah"],
        "Non-binary": ["Haru", "Sora", "Ren", "Kai", "Yuki", "Aki"],
    },
    "West Asian": {
        "Male": ["Dariush", "Reza", "Ali", "Arash", "Farhad", "Babak", "Amir", "Mehdi",
                  "Kamran", "Siavash", "Payam", "Behnam", "Nima", "Arman", "Kourosh"],
        "Female": ["Maryam", "Sara", "Niloufar", "Shirin", "Parisa", "Azadeh", "Leila",
                    "Nasrin", "Fatemeh", "Zahra", "Bahar", "Setareh", "Golnar", "Laleh"],
        "Non-binary": ["Azar", "Shirin", "Arya", "Dana", "Daria", "Noor"],
    },
}

LAST_NAMES = {
    "European": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Wilson", "Anderson",
                  "Taylor", "Thomas", "White", "Martin", "Thompson", "Campbell", "Robinson",
                  "Clark", "Lewis", "Hall", "Young", "King", "Scott", "Green", "Baker",
                  "Adams", "Nelson", "Mitchell", "Roberts", "Turner", "Phillips", "Stewart",
                  "MacDonald", "MacLeod", "Fraser", "Murray", "Ross", "Morrison", "Gordon",
                  "Reid", "Grant", "Henderson", "Hamilton", "Walker", "Wright", "Harris",
                  "Bell", "Duncan", "Ferguson", "McKay", "Sinclair", "Marshall"],
    "French Canadian": ["Tremblay", "Gagnon", "Roy", "Côté", "Bouchard", "Gauthier", "Morin",
                         "Lavoie", "Fortin", "Gagné", "Ouellet", "Pelletier", "Bélanger",
                         "Lévesque", "Bergeron", "Leblanc", "Paquette", "Girard", "Simard",
                         "Boucher", "Caron", "Beaulieu", "Cloutier", "Dubé", "Poirier",
                         "Fontaine", "Thibault", "Nadeau", "Charron", "Deschênes",
                         "Langlois", "Proulx", "Leclerc", "Arsenault", "Goulet"],
    "South Asian": ["Patel", "Singh", "Sharma", "Kumar", "Kaur", "Gupta", "Reddy", "Shah",
                     "Mehta", "Jain", "Desai", "Bhat", "Nair", "Gill", "Dhillon", "Sidhu",
                     "Grewal", "Sandhu", "Bajwa", "Cheema", "Chopra", "Banerjee", "Srinivasan",
                     "Verma", "Malhotra", "Khanna", "Kapoor", "Agarwal", "Bedi", "Johal"],
    "Chinese": ["Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao", "Wu", "Zhou",
                 "Xu", "Sun", "Ma", "Zhu", "Lin", "Hu", "Guo", "He", "Luo", "Liang",
                 "Deng", "Xiao", "Tang", "Zheng", "Feng", "Jiang", "Lu", "Pan", "Yao", "Cheng"],
    "Black": ["Williams", "Johnson", "Brown", "Jackson", "Thompson", "Harris", "Robinson",
               "Campbell", "Stewart", "Clarke", "Edwards", "Lewis", "Walker", "Morgan",
               "Pierre", "Jean-Baptiste", "Okafor", "Adeyemi", "Mensah", "Diallo",
               "Traoré", "Toure", "Abdi", "Hassan", "Baptiste", "Charles", "Denis",
               "Auguste", "Beaumont", "Okeke"],
    "Filipino": ["Santos", "Reyes", "Cruz", "Garcia", "Hernandez", "Gonzales", "Lopez",
                  "Martinez", "Rodriguez", "Torres", "Flores", "Ramirez", "Rivera", "Diaz",
                  "Mendoza", "Aquino", "Bautista", "Villanueva", "del Rosario", "Ramos"],
    "Arab": ["Ahmed", "Hassan", "Ali", "Mohamed", "Ibrahim", "Khalil", "Rahman", "Haddad",
              "Mansour", "Nasser", "Khoury", "Habib", "Farah", "Issa", "Saleh",
              "El-Sayed", "Abdallah", "Barakat", "Dabbagh", "Jaber"],
    "Latin American": ["Garcia", "Rodriguez", "Martinez", "Lopez", "Gonzalez", "Hernandez",
                        "Perez", "Sanchez", "Torres", "Rivera", "Flores", "Ramirez",
                        "Cruz", "Morales", "Ortiz", "Diaz", "Reyes", "Mendoza",
                        "Gutierrez", "Vargas"],
    "Southeast Asian": ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Phan", "Vu", "Vo",
                          "Dang", "Bui", "Do", "Ho", "Ngo", "Duong", "Ly",
                          "Soth", "Phon", "Chea", "Sok", "Chan"],
    "Indigenous": ["Bear", "Cardinal", "Flett", "Moose", "Clearsky", "Littlechild", "Sinclair",
                    "McKay", "Fontaine", "Thomas", "Anderson", "Bird", "Harper", "Daniels",
                    "Cook", "Fiddler", "Constant", "Mercredi", "Apetagon", "Blacksmith",
                    "Whitebear", "Brightnose", "Keeper", "Linklater", "Favel"],
    "Korean": ["Kim", "Lee", "Park", "Choi", "Jung", "Kang", "Cho", "Yoon", "Jang",
                "Lim", "Han", "Oh", "Shin", "Seo", "Kwon", "Hwang", "Song", "Ahn"],
    "Japanese": ["Tanaka", "Suzuki", "Takahashi", "Watanabe", "Ito", "Yamamoto", "Nakamura",
                  "Kobayashi", "Saito", "Kato", "Yoshida", "Yamada", "Sasaki", "Yamaguchi",
                  "Matsumoto", "Inoue", "Kimura", "Shimizu", "Hayashi", "Mori"],
    "West Asian": ["Tehrani", "Ahmadi", "Hosseini", "Rezaei", "Mousavi", "Hashemi",
                    "Moradi", "Karimi", "Ghorbani", "Eskandari", "Shirazi", "Bakhtiari",
                    "Farahani", "Javadi", "Kiani", "Nazari", "Rahimi", "Sadeghi"],
}

# ============================================================
# INVESTMENT / FINANCIAL ATTRIBUTES
# ============================================================

RISK_TOLERANCE = ["Very Conservative", "Conservative", "Moderate", "Growth", "Aggressive"]

INVESTMENT_GOALS = [
    "Retirement savings", "Wealth building", "Children's education (RESP)",
    "Home purchase down payment", "Income generation", "Emergency fund growth",
    "Business investment", "Estate/legacy planning", "Debt reduction then investing",
    "Early retirement / FIRE", "Passive income",
]

INVESTMENT_VEHICLES_KNOWN = [
    "Savings account", "GICs", "TFSA", "RRSP", "RESP", "Mutual funds",
    "ETFs", "Individual stocks", "Bonds", "Real estate", "Cryptocurrency",
    "FHSA", "Robo-advisors",
]

FINANCIAL_CONCERNS = [
    "Inflation eroding savings", "Market volatility", "Not saving enough for retirement",
    "Rising cost of living", "Housing affordability", "Healthcare costs in retirement",
    "Job security", "Managing debt", "Tax optimization", "Leaving inheritance",
    "Outliving savings", "Interest rate changes", "Not understanding investments",
    "Paying too much in fees", "Economic recession",
]

VALUES = [
    "Financial security", "Family well-being", "Environmental sustainability",
    "Community involvement", "Work-life balance", "Career advancement",
    "Education and learning", "Health and wellness", "Travel and experiences",
    "Homeownership", "Entrepreneurship", "Social justice",
    "Cultural preservation", "Religious/spiritual values", "Innovation and technology",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def weighted_choice(options_weights):
    """Choose from list of (option, weight) tuples."""
    options, weights = zip(*options_weights)
    return random.choices(options, weights=weights, k=1)[0]


def get_ethnicity_for_province(province):
    """Get ethnicity distribution adjusted for province."""
    mods = ETHNICITY_PROVINCE_MOD.get(province, {})
    adjusted = {}
    for eth, base_w in ETHNICITY_NATIONAL.items():
        adjusted[eth] = base_w * mods.get(eth, 1.0)
    # In Quebec, split European into French Canadian and European
    if province == "Quebec":
        fr_share = adjusted["European"] * 0.70
        eu_share = adjusted["European"] * 0.30
        del adjusted["European"]
        adjusted["French Canadian"] = fr_share
        adjusted["European"] = eu_share
    elif province == "New Brunswick":
        fr_share = adjusted["European"] * 0.30
        eu_share = adjusted["European"] * 0.70
        del adjusted["European"]
        adjusted["French Canadian"] = fr_share
        adjusted["European"] = eu_share
    elif province in ("Ontario", "Manitoba"):
        fr_share = adjusted["European"] * 0.04
        eu_share = adjusted["European"] * 0.96
        del adjusted["European"]
        adjusted["French Canadian"] = fr_share
        adjusted["European"] = eu_share
    else:
        adjusted["French Canadian"] = adjusted.get("European", 0) * 0.02
        adjusted["European"] = adjusted.get("European", 0) * 0.98

    items = list(adjusted.items())
    return weighted_choice([(k, v) for k, v in items])


def get_education(age):
    """Education level conditional on age."""
    ed_weights = list(EDUCATION_LEVELS)
    if age < 25:
        # Younger people less likely to have graduate degrees
        ed_weights = [
            ("No certificate/diploma", 5),
            ("High school diploma", 40),
            ("College diploma or trades certificate", 35),
            ("Bachelor's degree", 18),
            ("Master's degree", 2),
            ("Doctorate or professional degree", 0),
        ]
    elif age < 30:
        ed_weights = [
            ("No certificate/diploma", 5),
            ("High school diploma", 22),
            ("College diploma or trades certificate", 33),
            ("Bachelor's degree", 28),
            ("Master's degree", 10),
            ("Doctorate or professional degree", 2),
        ]
    return weighted_choice(ed_weights)


def get_income(age, education, province):
    """Household income based on education, age, and province."""
    # Base income by education
    base_income = {
        "No certificate/diploma": 38000,
        "High school diploma": 52000,
        "College diploma or trades certificate": 65000,
        "Bachelor's degree": 82000,
        "Master's degree": 100000,
        "Doctorate or professional degree": 120000,
    }
    income = base_income.get(education, 55000)

    # Age curve (peaks 45-55)
    if age < 25:
        income *= 0.55
    elif age < 30:
        income *= 0.72
    elif age < 35:
        income *= 0.85
    elif age < 40:
        income *= 0.95
    elif age < 50:
        income *= 1.10
    elif age < 55:
        income *= 1.15
    elif age < 60:
        income *= 1.05
    elif age < 65:
        income *= 0.90
    elif age < 70:
        income *= 0.70  # retirement income
    else:
        income *= 0.60

    # Province modifier
    province_mod = {
        "Ontario": 1.08, "Quebec": 0.92, "British Columbia": 1.05,
        "Alberta": 1.15, "Manitoba": 0.90, "Saskatchewan": 0.92,
        "Nova Scotia": 0.85, "New Brunswick": 0.83,
        "Newfoundland and Labrador": 0.88, "Prince Edward Island": 0.82,
        "Northwest Territories": 1.25, "Yukon": 1.15, "Nunavut": 1.20,
    }
    income *= province_mod.get(province, 1.0)

    # Random variation (±30%)
    income *= random.uniform(0.70, 1.35)

    # Add spouse income for some (dual income households)
    if age >= 25 and random.random() < 0.55:
        spouse_contribution = income * random.uniform(0.3, 0.9)
        income += spouse_contribution

    return max(25000, round(income / 1000) * 1000)


def get_net_worth(age, income, province, home_owner, home_value):
    """Estimate net worth based on age, income, home ownership."""
    # Savings accumulation model
    years_earning = max(0, age - 22)
    avg_savings_rate = 0.08 + random.uniform(-0.04, 0.08)
    financial_assets = years_earning * income * avg_savings_rate * random.uniform(0.5, 1.8)

    # Investment growth factor
    growth = 1 + (years_earning * 0.03)
    financial_assets *= min(growth, 4.0)

    # Home equity
    home_equity = 0
    if home_owner:
        mortgage_paid_pct = min(0.95, max(0.1, (age - 28) * 0.025 + random.uniform(-0.1, 0.1)))
        home_equity = home_value * mortgage_paid_pct

    # Pension value estimate for older workers
    pension_value = 0
    if age >= 40:
        pension_value = (age - 40) * income * 0.03

    net_worth = financial_assets + home_equity + pension_value

    # Add debt reduction for younger people
    if age < 35:
        student_debt = random.choice([0, 0, 0, 15000, 25000, 35000, 50000])
        net_worth -= student_debt

    return max(-50000, round(net_worth / 1000) * 1000)


def get_employment(age, education):
    """Employment status based on age and education."""
    if age >= 65:
        return weighted_choice([
            ("Retired", 65), ("Employed part-time", 12), ("Self-employed", 10),
            ("Employed full-time", 8), ("Unemployed", 5),
        ])
    elif age >= 55:
        return weighted_choice([
            ("Employed full-time", 50), ("Self-employed", 15), ("Retired", 15),
            ("Employed part-time", 10), ("Unemployed", 7), ("Student", 3),
        ])
    elif age < 25:
        return weighted_choice([
            ("Student", 35), ("Employed full-time", 30), ("Employed part-time", 20),
            ("Unemployed", 10), ("Self-employed", 5),
        ])
    else:
        return weighted_choice([
            ("Employed full-time", 62), ("Self-employed", 12),
            ("Employed part-time", 10), ("Unemployed", 6), ("Student", 5),
            ("Retired", 5),
        ])


def get_occupation(education, employment):
    """Get occupation category."""
    if employment in ("Retired", "Unemployed", "Student"):
        return employment

    if education in ("Master's degree", "Doctorate or professional degree"):
        return weighted_choice([
            ("Sciences/Technology/Engineering", 18), ("Health", 18),
            ("Education/Law/Government", 20), ("Management", 18),
            ("Business/Finance/Administration", 15), ("Information Technology", 8),
            ("Arts/Culture/Recreation/Sport", 3),
        ])
    elif education == "Bachelor's degree":
        return weighted_choice([
            ("Business/Finance/Administration", 20), ("Management", 12),
            ("Sciences/Technology/Engineering", 12), ("Education/Law/Government", 12),
            ("Information Technology", 12), ("Health", 10),
            ("Sales and Service", 10), ("Arts/Culture/Recreation/Sport", 5),
            ("Trades/Transport/Equipment", 4), ("Natural Resources/Agriculture", 3),
        ])
    elif education == "College diploma or trades certificate":
        return weighted_choice([
            ("Trades/Transport/Equipment", 22), ("Sales and Service", 18),
            ("Health", 12), ("Business/Finance/Administration", 12),
            ("Manufacturing/Utilities", 8), ("Natural Resources/Agriculture", 6),
            ("Management", 6), ("Information Technology", 5),
            ("Sciences/Technology/Engineering", 5), ("Education/Law/Government", 4),
            ("Arts/Culture/Recreation/Sport", 2),
        ])
    else:
        return weighted_choice([
            ("Sales and Service", 30), ("Trades/Transport/Equipment", 20),
            ("Manufacturing/Utilities", 12), ("Natural Resources/Agriculture", 10),
            ("Business/Finance/Administration", 10), ("Management", 5),
            ("Health", 5), ("Arts/Culture/Recreation/Sport", 3),
            ("Education/Law/Government", 3), ("Information Technology", 2),
        ])


def get_family_status(age, gender):
    """Family status based on age."""
    if age < 25:
        return weighted_choice([("Single", 70), ("Common-law partner", 20), ("Married", 8), ("Divorced", 2)])
    elif age < 35:
        return weighted_choice([("Single", 30), ("Common-law partner", 20), ("Married", 42), ("Divorced", 8)])
    elif age < 50:
        return weighted_choice([("Single", 12), ("Common-law partner", 10), ("Married", 55), ("Divorced", 18), ("Separated", 5)])
    elif age < 65:
        return weighted_choice([("Single", 10), ("Married", 55), ("Divorced", 20), ("Widowed", 8), ("Common-law partner", 7)])
    else:
        return weighted_choice([("Married", 45), ("Widowed", 25), ("Divorced", 15), ("Single", 10), ("Common-law partner", 5)])


def get_dependents(age, family_status):
    """Number of dependents."""
    if family_status == "Single" and age < 30:
        return weighted_choice([(0, 85), (1, 10), (2, 5)])
    elif age < 25:
        return weighted_choice([(0, 70), (1, 20), (2, 10)])
    elif age < 35:
        return weighted_choice([(0, 35), (1, 30), (2, 25), (3, 8), (4, 2)])
    elif age < 50:
        return weighted_choice([(0, 15), (1, 20), (2, 35), (3, 20), (4, 8), (5, 2)])
    elif age < 60:
        return weighted_choice([(0, 40), (1, 25), (2, 25), (3, 8), (4, 2)])
    else:
        return weighted_choice([(0, 75), (1, 15), (2, 8), (3, 2)])


def get_risk_tolerance(age, income, net_worth, education):
    """Risk tolerance based on demographics."""
    if age >= 65:
        weights = [("Very Conservative", 25), ("Conservative", 35), ("Moderate", 25), ("Growth", 12), ("Aggressive", 3)]
    elif age >= 50:
        weights = [("Very Conservative", 10), ("Conservative", 25), ("Moderate", 35), ("Growth", 22), ("Aggressive", 8)]
    elif age >= 35:
        weights = [("Very Conservative", 5), ("Conservative", 15), ("Moderate", 35), ("Growth", 30), ("Aggressive", 15)]
    else:
        weights = [("Very Conservative", 5), ("Conservative", 10), ("Moderate", 25), ("Growth", 35), ("Aggressive", 25)]

    # Higher income/education slightly shifts toward growth
    if income > 150000 or education in ("Master's degree", "Doctorate or professional degree"):
        weights = [(r, w * (1.3 if r in ("Growth", "Aggressive") else 0.85)) for r, w in weights]

    return weighted_choice(weights)


def get_investment_experience(age):
    """Years of investment experience."""
    max_possible = max(0, age - 20)
    if max_possible == 0:
        return 0
    # Many Canadians start investing late
    exp = random.choice([
        0, 0, 0,  # ~30% no experience
        random.randint(1, 3),  # beginner
        random.randint(1, 3),
        random.randint(3, 8),  # intermediate
        random.randint(3, 8),
        random.randint(8, 15),  # experienced
        random.randint(15, max(15, max_possible)),  # veteran
    ])
    return min(exp, max_possible)


def get_investment_knowledge(experience, education):
    """Self-assessed investment knowledge."""
    if experience == 0:
        return weighted_choice([("Beginner", 70), ("Some knowledge", 25), ("Knowledgeable", 5)])
    elif experience < 5:
        return weighted_choice([("Beginner", 20), ("Some knowledge", 50), ("Knowledgeable", 25), ("Advanced", 5)])
    elif experience < 15:
        return weighted_choice([("Some knowledge", 15), ("Knowledgeable", 45), ("Advanced", 35), ("Expert", 5)])
    else:
        return weighted_choice([("Knowledgeable", 15), ("Advanced", 45), ("Expert", 40)])


def get_current_investments(experience, income, age, net_worth):
    """What investment vehicles the person currently uses."""
    investments = []
    if income > 0 or net_worth > 0:
        # Almost everyone has a savings account
        if random.random() < 0.90:
            investments.append("Savings account")

    if experience == 0:
        return investments if investments else ["Savings account"]

    # TFSA is very common
    if random.random() < 0.70:
        investments.append("TFSA")
    # RRSP
    if age >= 25 and random.random() < 0.55:
        investments.append("RRSP")
    # GICs
    if random.random() < 0.25 + (0.15 if age > 55 else 0):
        investments.append("GICs")
    # Mutual funds
    if random.random() < 0.40:
        investments.append("Mutual funds")
    # ETFs
    if experience >= 2 and random.random() < 0.35:
        investments.append("ETFs")
    # Individual stocks
    if experience >= 3 and random.random() < 0.25:
        investments.append("Individual stocks")
    # Bonds
    if experience >= 2 and random.random() < 0.15:
        investments.append("Bonds")
    # RESP
    if age >= 28 and age <= 60 and random.random() < 0.20:
        investments.append("RESP")
    # Crypto
    if age < 50 and random.random() < 0.15:
        investments.append("Cryptocurrency")
    # FHSA
    if 18 <= age <= 40 and random.random() < 0.10:
        investments.append("FHSA")
    # Robo-advisor
    if age < 45 and random.random() < 0.12:
        investments.append("Robo-advisors")
    # Real estate investment
    if net_worth > 300000 and random.random() < 0.10:
        investments.append("Real estate investment")

    return investments if investments else ["Savings account"]


def get_primary_language(province, ethnicity):
    """Primary language based on province and ethnicity."""
    if ethnicity == "French Canadian":
        return "French"
    if province == "Quebec":
        if random.random() < 0.78:
            return "French"
        return random.choice(["English", "English and French"])
    if province == "New Brunswick":
        if random.random() < 0.30:
            return "French"
        return "English"
    if ethnicity in ("Chinese", "South Asian", "Filipino", "Arab", "Korean",
                     "Japanese", "Southeast Asian", "Latin American", "West Asian"):
        r = random.random()
        if r < 0.55:
            return "English"
        elif r < 0.80:
            return "English and heritage language"
        else:
            return "Heritage language and English"
    return "English"


def get_tech_savviness(age, education):
    """Tech savviness level."""
    if age < 30:
        return weighted_choice([("Low", 5), ("Medium", 25), ("High", 45), ("Very High", 25)])
    elif age < 45:
        return weighted_choice([("Low", 8), ("Medium", 30), ("High", 40), ("Very High", 22)])
    elif age < 60:
        return weighted_choice([("Low", 15), ("Medium", 40), ("High", 35), ("Very High", 10)])
    else:
        return weighted_choice([("Low", 30), ("Medium", 40), ("High", 25), ("Very High", 5)])


def get_urban_rural(city, province):
    """Classify as urban, suburban, or rural."""
    major_cities = {"Toronto", "Montreal", "Vancouver", "Calgary", "Edmonton", "Ottawa",
                    "Winnipeg", "Quebec City", "Hamilton", "Halifax", "Victoria"}
    if city in major_cities:
        return weighted_choice([("Urban", 55), ("Suburban", 40), ("Rural", 5)])
    else:
        return weighted_choice([("Urban", 25), ("Suburban", 35), ("Rural", 40)])


# ============================================================
# PERSONA GENERATION
# ============================================================

def generate_persona(persona_id):
    """Generate a single synthetic persona."""
    # 1. Geography
    province = weighted_choice(list(PROVINCE_WEIGHTS.items()))
    city_options = CITIES_BY_PROVINCE[province]
    city = weighted_choice(city_options)
    urban_rural = get_urban_rural(city, province)

    # 2. Demographics
    age_bracket = weighted_choice(AGE_BRACKETS)
    age = random.randint(age_bracket[0], age_bracket[1])
    gender = weighted_choice(GENDERS)
    ethnicity = get_ethnicity_for_province(province)

    # 3. Name
    name_eth = ethnicity if ethnicity in FIRST_NAMES else "European"
    gender_key = gender if gender in FIRST_NAMES[name_eth] else "Non-binary"
    first_name = random.choice(FIRST_NAMES[name_eth][gender_key])
    last_name = random.choice(LAST_NAMES[name_eth])

    # 4. Language
    primary_language = get_primary_language(province, ethnicity)

    # 5. Education & Employment
    education = get_education(age)
    employment = get_employment(age, education)
    occupation = get_occupation(education, employment)

    # 6. Family
    family_status = get_family_status(age, gender)
    num_dependents = get_dependents(age, family_status)

    # 7. Financials
    household_income = get_income(age, education, province)

    # Home ownership
    age_bracket_key = None
    for bracket, rate in HOME_OWNERSHIP_BY_AGE.items():
        if bracket[0] <= age <= bracket[1]:
            age_bracket_key = bracket
            break
    if age_bracket_key is None:
        age_bracket_key = (75, 85)
    home_owner = random.random() < HOME_OWNERSHIP_BY_AGE[age_bracket_key]

    home_value = 0
    mortgage_remaining = 0
    if home_owner:
        base_price = HOME_PRICE_BY_PROVINCE.get(province, 400000)
        city_mult = CITY_PREMIUM.get(city, 1.0)
        home_value = round(base_price * city_mult * random.uniform(0.6, 1.6) / 10000) * 10000
        # Mortgage remaining
        years_owned = min(age - 25, random.randint(1, 25))
        if years_owned > 0:
            mortgage_remaining = max(0, round(home_value * max(0, (1 - years_owned * 0.04)) * random.uniform(0.3, 1.0) / 5000) * 5000)
        else:
            mortgage_remaining = round(home_value * 0.80 / 5000) * 5000

    net_worth = get_net_worth(age, household_income, province, home_owner, home_value)

    # Investable assets (subset of net worth)
    if net_worth > 0:
        investable_pct = random.uniform(0.15, 0.60)
        investable_assets = round(net_worth * investable_pct / 1000) * 1000
    else:
        investable_assets = random.choice([0, 1000, 2000, 5000, 10000])

    # 8. Investment profile
    investment_experience = get_investment_experience(age)
    investment_knowledge = get_investment_knowledge(investment_experience, education)
    risk_tolerance = get_risk_tolerance(age, household_income, net_worth, education)
    current_investments = get_current_investments(investment_experience, household_income, age, net_worth)

    # Investment goals (pick 2-3)
    possible_goals = list(INVESTMENT_GOALS)
    if num_dependents > 0 and age < 55:
        possible_goals.append("Children's education (RESP)")  # extra weight
    if not home_owner and age < 50:
        possible_goals.append("Home purchase down payment")  # extra weight
    if age >= 50:
        possible_goals.extend(["Retirement savings", "Retirement savings"])  # extra weight
    num_goals = random.randint(2, 3)
    investment_goals = random.sample(possible_goals, min(num_goals, len(possible_goals)))
    # Deduplicate
    investment_goals = list(dict.fromkeys(investment_goals))

    # Investment timeline
    if age >= 60:
        timeline = weighted_choice([("Short-term (1-3 years)", 30), ("Medium-term (3-10 years)", 45), ("Long-term (10+ years)", 25)])
    elif age >= 45:
        timeline = weighted_choice([("Short-term (1-3 years)", 15), ("Medium-term (3-10 years)", 40), ("Long-term (10+ years)", 45)])
    else:
        timeline = weighted_choice([("Short-term (1-3 years)", 15), ("Medium-term (3-10 years)", 30), ("Long-term (10+ years)", 55)])

    # Financial concerns (pick 2-3)
    num_concerns = random.randint(2, 3)
    financial_concerns = random.sample(FINANCIAL_CONCERNS, num_concerns)

    # 9. Psychographics
    tech_savvy = get_tech_savviness(age, education)
    top_values = random.sample(VALUES, 3)

    # Preferred investment channel
    if tech_savvy in ("High", "Very High") and age < 50:
        pref_channel = weighted_choice([
            ("Online self-directed", 40), ("Robo-advisor", 20),
            ("Financial advisor", 25), ("Bank advisor", 15),
        ])
    elif age >= 60:
        pref_channel = weighted_choice([
            ("Financial advisor", 40), ("Bank advisor", 30),
            ("Online self-directed", 20), ("Robo-advisor", 10),
        ])
    else:
        pref_channel = weighted_choice([
            ("Financial advisor", 30), ("Online self-directed", 25),
            ("Bank advisor", 25), ("Robo-advisor", 20),
        ])

    # 10. Life stage
    if age < 25:
        life_stage = "Early career / Student"
    elif age < 35:
        if num_dependents > 0:
            life_stage = "Young family"
        else:
            life_stage = "Establishing career"
    elif age < 50:
        if num_dependents > 0:
            life_stage = "Growing family / Mid-career"
        else:
            life_stage = "Mid-career"
    elif age < 60:
        life_stage = "Peak earning years"
    elif age < 68:
        life_stage = "Pre-retirement / Early retirement"
    else:
        life_stage = "Retirement"

    # 11. Build narrative summary
    pronoun_subj = "He" if gender == "Male" else ("She" if gender == "Female" else "They")
    pronoun_poss = "his" if gender == "Male" else ("her" if gender == "Female" else "their")
    pronoun_obj = "him" if gender == "Male" else ("her" if gender == "Female" else "them")
    verb_s = "s" if gender != "Non-binary" else ""
    verb_is = "is" if gender != "Non-binary" else "are"
    verb_has = "has" if gender != "Non-binary" else "have"

    # Employment description
    if employment == "Retired":
        emp_desc = f"{pronoun_subj} {verb_is} retired"
        if occupation != "Retired":
            emp_desc += f" (formerly in {occupation.lower()})"
    elif employment == "Student":
        emp_desc = f"{pronoun_subj} {verb_is} a student"
    elif employment == "Unemployed":
        emp_desc = f"{pronoun_subj} {verb_is} currently between jobs"
    else:
        emp_desc = f"{pronoun_subj} work{verb_s} {employment.lower().replace('employed ', '')} in {occupation.lower()}"

    # Family description
    if family_status in ("Married", "Common-law partner"):
        if num_dependents > 0:
            fam_desc = f"{pronoun_subj} {verb_is} {family_status.lower()} with {num_dependents} dependent{'s' if num_dependents > 1 else ''}"
        else:
            fam_desc = f"{pronoun_subj} {verb_is} {family_status.lower()} with no dependents"
    elif family_status == "Divorced":
        fam_desc = f"{pronoun_subj} {verb_is} divorced" + (f" with {num_dependents} dependent{'s' if num_dependents > 1 else ''}" if num_dependents > 0 else "")
    elif family_status == "Widowed":
        fam_desc = f"{pronoun_subj} {verb_is} widowed" + (f" with {num_dependents} dependent{'s' if num_dependents > 1 else ''}" if num_dependents > 0 else "")
    else:
        fam_desc = f"{pronoun_subj} {verb_is} single" + (f" with {num_dependents} dependent{'s' if num_dependents > 1 else ''}" if num_dependents > 0 else "")

    # Home description
    if home_owner:
        home_desc = f"own{verb_s} {pronoun_poss} home (valued at ~${home_value:,})"
        if mortgage_remaining > 0:
            home_desc += f" with ${mortgage_remaining:,} remaining on the mortgage"
    else:
        home_desc = f"rent{verb_s} {pronoun_poss} home"

    # Investment description
    if investment_experience == 0:
        inv_desc = f"{pronoun_subj} {verb_has} no prior investment experience and {verb_is} looking to start investing"
    elif investment_experience < 5:
        inv_desc = f"{pronoun_subj} {verb_has} {investment_experience} year{'s' if investment_experience > 1 else ''} of investment experience ({investment_knowledge.lower()} level)"
    else:
        inv_desc = f"{pronoun_subj} {verb_has} {investment_experience} years of investment experience ({investment_knowledge.lower()} level)"

    current_inv_str = ", ".join(current_investments) if current_investments else "none yet"

    narrative = (
        f"{first_name} {last_name} is a {age}-year-old {ethnicity} {gender.lower()} "
        f"living in {city}, {province}. "
        f"{pronoun_subj} {verb_has} a {education.lower()} and speak{verb_s} {primary_language.lower()}. "
        f"{emp_desc}, with a household income of ~${household_income:,}/year. "
        f"{fam_desc}. {pronoun_subj} {home_desc}. "
        f"{pronoun_poss.capitalize()} estimated net worth is ${net_worth:,} with ~${investable_assets:,} in investable assets. "
        f"{inv_desc}. Current investments include: {current_inv_str}. "
        f"{pronoun_subj} {verb_has} a {risk_tolerance.lower()} risk tolerance and prefer{verb_s} {pref_channel.lower()} for investing. "
        f"{pronoun_poss.capitalize()} primary investment goals are: {', '.join(investment_goals).lower()}. "
        f"Top concerns: {', '.join(financial_concerns).lower()}. "
        f"Values: {', '.join(top_values).lower()}. "
        f"Tech savviness: {tech_savvy.lower()}. Life stage: {life_stage.lower()}."
    )

    return {
        "id": persona_id + 1,
        "first_name": first_name,
        "last_name": last_name,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "province": province,
        "city": city,
        "urban_rural": urban_rural,
        "primary_language": primary_language,
        "education": education,
        "employment_status": employment,
        "occupation": occupation,
        "family_status": family_status,
        "num_dependents": num_dependents,
        "household_income": household_income,
        "home_owner": home_owner,
        "home_value": home_value if home_owner else 0,
        "mortgage_remaining": mortgage_remaining if home_owner else 0,
        "net_worth": net_worth,
        "investable_assets": investable_assets,
        "investment_experience_years": investment_experience,
        "investment_knowledge": investment_knowledge,
        "risk_tolerance": risk_tolerance,
        "current_investments": current_investments,
        "investment_goals": investment_goals,
        "investment_timeline": timeline,
        "preferred_channel": pref_channel,
        "financial_concerns": financial_concerns,
        "tech_savviness": tech_savvy,
        "top_values": top_values,
        "life_stage": life_stage,
        "persona_summary": narrative,
    }


# ============================================================
# OUTPUT AND VERIFICATION
# ============================================================

def write_csv(personas, filepath):
    """Write personas to CSV."""
    flat_fields = [
        "id", "first_name", "last_name", "age", "gender", "ethnicity",
        "province", "city", "urban_rural", "primary_language", "education",
        "employment_status", "occupation", "family_status", "num_dependents",
        "household_income", "home_owner", "home_value", "mortgage_remaining",
        "net_worth", "investable_assets", "investment_experience_years",
        "investment_knowledge", "risk_tolerance", "current_investments",
        "investment_goals", "investment_timeline", "preferred_channel",
        "financial_concerns", "tech_savviness", "top_values", "life_stage",
        "persona_summary",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for p in personas:
            row = dict(p)
            # Flatten lists to semicolon-separated strings
            for key in ("current_investments", "investment_goals", "financial_concerns", "top_values"):
                if isinstance(row[key], list):
                    row[key] = "; ".join(row[key])
            writer.writerow(row)


def write_json(personas, filepath):
    """Write personas to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)


def print_summary(personas):
    """Print demographic summary for verification."""
    from collections import Counter

    print(f"\n{'='*60}")
    print(f"  SYNTHETIC PERSONA GENERATION SUMMARY")
    print(f"  Total personas: {len(personas)}")
    print(f"{'='*60}")

    def show_dist(title, key):
        counts = Counter(p[key] for p in personas)
        total = len(personas)
        print(f"\n{title}:")
        for item, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "#" * int(pct / 2)
            print(f"  {item:40s} {count:4d} ({pct:5.1f}%) {bar}")

    show_dist("PROVINCE", "province")
    show_dist("GENDER", "gender")
    show_dist("ETHNICITY", "ethnicity")
    show_dist("EDUCATION", "education")
    show_dist("EMPLOYMENT STATUS", "employment_status")
    show_dist("RISK TOLERANCE", "risk_tolerance")
    show_dist("INVESTMENT KNOWLEDGE", "investment_knowledge")
    show_dist("LIFE STAGE", "life_stage")
    show_dist("FAMILY STATUS", "family_status")
    show_dist("PREFERRED CHANNEL", "preferred_channel")

    # Age distribution
    print(f"\nAGE DISTRIBUTION:")
    age_brackets = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 74), (75, 85)]
    for lo, hi in age_brackets:
        count = sum(1 for p in personas if lo <= p["age"] <= hi)
        pct = count / len(personas) * 100
        bar = "#" * int(pct / 2)
        print(f"  {lo}-{hi:2d}:  {count:4d} ({pct:5.1f}%) {bar}")

    # Income distribution
    print(f"\nHOUSEHOLD INCOME DISTRIBUTION:")
    brackets = [(0, 30000), (30001, 60000), (60001, 100000), (100001, 150000), (150001, 250000), (250001, 999999999)]
    labels = ["Under $30K", "$30K-$60K", "$60K-$100K", "$100K-$150K", "$150K-$250K", "$250K+"]
    for (lo, hi), label in zip(brackets, labels):
        count = sum(1 for p in personas if lo <= p["household_income"] <= hi)
        pct = count / len(personas) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:20s} {count:4d} ({pct:5.1f}%) {bar}")

    # Home ownership
    owners = sum(1 for p in personas if p["home_owner"])
    print(f"\nHOME OWNERSHIP: {owners} ({owners/len(personas)*100:.1f}%)")

    # Net worth distribution
    print(f"\nNET WORTH DISTRIBUTION:")
    nw_brackets = [(-999999, 0), (0, 50000), (50001, 200000), (200001, 500000), (500001, 1000000), (1000001, 999999999)]
    nw_labels = ["Negative", "$0-$50K", "$50K-$200K", "$200K-$500K", "$500K-$1M", "$1M+"]
    for (lo, hi), label in zip(nw_brackets, nw_labels):
        count = sum(1 for p in personas if lo <= p["net_worth"] <= hi)
        pct = count / len(personas) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:20s} {count:4d} ({pct:5.1f}%) {bar}")

    # Average stats
    avg_age = sum(p["age"] for p in personas) / len(personas)
    avg_income = sum(p["household_income"] for p in personas) / len(personas)
    avg_nw = sum(p["net_worth"] for p in personas) / len(personas)
    median_income = sorted(p["household_income"] for p in personas)[len(personas)//2]
    median_nw = sorted(p["net_worth"] for p in personas)[len(personas)//2]

    print(f"\nKEY STATISTICS:")
    print(f"  Average age:           {avg_age:.1f}")
    print(f"  Average income:        ${avg_income:,.0f}")
    print(f"  Median income:         ${median_income:,}")
    print(f"  Average net worth:     ${avg_nw:,.0f}")
    print(f"  Median net worth:      ${median_nw:,}")


def main():
    random.seed(SEED)

    print("Generating 1,000 synthetic Canadian investor personas...")
    personas = [generate_persona(i) for i in range(NUM_PERSONAS)]

    # Write outputs
    csv_path = "personas.csv"
    json_path = "personas.json"

    write_csv(personas, csv_path)
    write_json(personas, json_path)

    print(f"\nFiles written:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")

    print_summary(personas)

    # Print a sample persona
    print(f"\n{'='*60}")
    print(f"  SAMPLE PERSONA (#{personas[0]['id']})")
    print(f"{'='*60}")
    print(personas[0]["persona_summary"])
    print(f"\n{'='*60}")
    print(f"  SAMPLE PERSONA (#{personas[500]['id']})")
    print(f"{'='*60}")
    print(personas[500]["persona_summary"])


if __name__ == "__main__":
    main()
