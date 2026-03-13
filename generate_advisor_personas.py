#!/usr/bin/env python3
"""
Canadian Financial Advisor Persona Generator
Generates 1000 demographically proportional synthetic personas of Canadian
financial advisors. Uses industry survey data and regulatory statistics.
"""

import random
import json
import csv
import math
from collections import Counter

SEED = 43
NUM_PERSONAS = 1000

# ============================================================
# GEOGRAPHIC DISTRIBUTIONS (advisors concentrated in financial centers)
# ============================================================

PROVINCE_WEIGHTS = {
    "Ontario": 40.0,
    "Quebec": 18.0,
    "British Columbia": 15.0,
    "Alberta": 12.0,
    "Manitoba": 4.0,
    "Saskatchewan": 3.0,
    "Nova Scotia": 3.0,
    "New Brunswick": 2.0,
    "Newfoundland and Labrador": 1.0,
    "Prince Edward Island": 0.5,
    "Northwest Territories": 0.5,
    "Yukon": 0.5,
    "Nunavut": 0.5,
}

# Major cities/towns by province — advisors cluster in larger cities
CITIES_BY_PROVINCE = {
    "Ontario": [
        ("Toronto", 45), ("Ottawa", 10), ("Mississauga", 6), ("Brampton", 3),
        ("Hamilton", 4), ("London", 4), ("Markham", 3), ("Kitchener", 3),
        ("Windsor", 2), ("Oshawa", 2), ("Barrie", 2), ("Oakville", 3),
        ("Burlington", 2), ("Richmond Hill", 2), ("Vaughan", 2),
        ("Waterloo", 2), ("Guelph", 2), ("Kingston", 1), ("St. Catharines", 2),
    ],
    "Quebec": [
        ("Montreal", 50), ("Quebec City", 15), ("Laval", 8), ("Gatineau", 6),
        ("Longueuil", 5), ("Sherbrooke", 4), ("Trois-Rivieres", 3),
        ("Levis", 3), ("Saguenay", 2), ("Terrebonne", 2), ("Drummondville", 2),
    ],
    "British Columbia": [
        ("Vancouver", 40), ("Surrey", 10), ("Burnaby", 8), ("Richmond", 6),
        ("Victoria", 10), ("Kelowna", 5), ("Coquitlam", 4), ("Abbotsford", 3),
        ("Nanaimo", 3), ("Kamloops", 3), ("North Vancouver", 4), ("Langley", 2),
        ("New Westminster", 2),
    ],
    "Alberta": [
        ("Calgary", 42), ("Edmonton", 38), ("Red Deer", 4), ("Lethbridge", 3),
        ("St. Albert", 3), ("Airdrie", 3), ("Sherwood Park", 3),
        ("Medicine Hat", 2), ("Grande Prairie", 2),
    ],
    "Manitoba": [
        ("Winnipeg", 75), ("Brandon", 8), ("Steinbach", 4), ("Thompson", 3),
        ("Portage la Prairie", 3), ("Selkirk", 3), ("Winkler", 2), ("Morden", 2),
    ],
    "Saskatchewan": [
        ("Saskatoon", 40), ("Regina", 40), ("Prince Albert", 5), ("Moose Jaw", 4),
        ("Swift Current", 3), ("Yorkton", 3), ("North Battleford", 3),
        ("Estevan", 2),
    ],
    "Nova Scotia": [
        ("Halifax", 65), ("Dartmouth", 10), ("Sydney", 6), ("Truro", 5),
        ("New Glasgow", 4), ("Kentville", 3), ("Bridgewater", 3),
        ("Antigonish", 2), ("Amherst", 2),
    ],
    "New Brunswick": [
        ("Moncton", 28), ("Saint John", 28), ("Fredericton", 22),
        ("Dieppe", 5), ("Miramichi", 5), ("Edmundston", 4),
        ("Bathurst", 4), ("Riverview", 4),
    ],
    "Newfoundland and Labrador": [
        ("St. John's", 60), ("Mount Pearl", 12), ("Corner Brook", 10),
        ("Conception Bay South", 8), ("Paradise", 5), ("Grand Falls-Windsor", 5),
    ],
    "Prince Edward Island": [
        ("Charlottetown", 60), ("Summerside", 20), ("Stratford", 10),
        ("Cornwall", 10),
    ],
    "Northwest Territories": [
        ("Yellowknife", 70), ("Hay River", 12), ("Inuvik", 10),
        ("Fort Smith", 8),
    ],
    "Yukon": [
        ("Whitehorse", 80), ("Dawson City", 10), ("Watson Lake", 10),
    ],
    "Nunavut": [
        ("Iqaluit", 50), ("Rankin Inlet", 20), ("Arviat", 15),
        ("Cambridge Bay", 15),
    ],
}

# ============================================================
# ADVISOR DEMOGRAPHIC DISTRIBUTIONS
# ============================================================

# Age brackets (advisors skew older — industry requires credentials + experience)
AGE_BRACKETS = [
    ((24, 29), 11),   # Under 30
    ((30, 34), 14),   # Early 30s
    ((35, 39), 14),   # Late 30s
    ((40, 44), 13),   # Early 40s
    ((45, 49), 12),   # Late 40s
    ((50, 54), 11),   # Early 50s
    ((55, 59), 8),    # Late 50s
    ((60, 65), 5),    # Early 60s
    ((66, 72), 2),    # Late career
]

GENDERS = [("Male", 82), ("Female", 17), ("Non-binary", 1)]

# Ethnicity — financial advisory skews European/South Asian more than general population
ETHNICITY_NATIONAL = {
    "European": 72,
    "South Asian": 8,
    "Chinese": 5,
    "Black": 2.5,
    "Filipino": 1.5,
    "Arab": 2,
    "Latin American": 1,
    "Southeast Asian": 0.5,
    "Indigenous": 2,
    "Korean": 1.5,
    "Japanese": 0.5,
    "West Asian": 2,
    "French Canadian": 2,
}

# Provincial ethnicity modifiers (same logic as consumer generator)
ETHNICITY_PROVINCE_MOD = {
    "Ontario": {"South Asian": 1.5, "Chinese": 1.2, "Black": 1.3, "Arab": 1.2, "Indigenous": 0.4},
    "Quebec": {"European": 0.6, "French Canadian": 8.0, "Arab": 2.0, "Black": 1.2, "South Asian": 0.5, "Chinese": 0.6, "Indigenous": 0.4},
    "British Columbia": {"Chinese": 2.2, "South Asian": 1.6, "Korean": 1.5, "Japanese": 2.0, "Indigenous": 0.6, "Black": 0.3},
    "Alberta": {"South Asian": 1.2, "Filipino": 1.4, "Chinese": 0.9, "Indigenous": 0.8},
    "Manitoba": {"Indigenous": 2.5, "Filipino": 2.0, "South Asian": 0.7},
    "Saskatchewan": {"Indigenous": 2.8, "South Asian": 0.6, "Filipino": 0.8},
    "Nova Scotia": {"Black": 1.5, "Indigenous": 1.3, "Arab": 1.2},
    "New Brunswick": {"French Canadian": 4.0, "Indigenous": 0.8},
    "Newfoundland and Labrador": {"European": 1.1, "Indigenous": 0.8},
    "Prince Edward Island": {"European": 1.1},
    "Northwest Territories": {"Indigenous": 5.0, "European": 0.7},
    "Yukon": {"Indigenous": 3.5, "European": 0.8},
    "Nunavut": {"Indigenous": 10.0, "European": 0.2},
}

# Education (advisors are more educated)
EDUCATION_LEVELS = [
    ("College diploma or trades certificate", 10),
    ("Bachelor's degree", 55),
    ("Master's degree", 30),
    ("Doctorate or professional degree", 5),
]

# ============================================================
# ADVISOR-SPECIFIC ATTRIBUTE DISTRIBUTIONS
# ============================================================

FIRM_TYPES = [
    ("Bank-owned brokerage", 30),
    ("Bank retail branch", 20),
    ("Independent dealer", 20),
    ("Insurance-based firm", 15),
    ("Independent RIA", 10),
    ("Boutique / Family office", 5),
]

PRACTICE_FOCUS_OPTIONS = [
    ("Retirement Planning", 25),
    ("Wealth Management", 20),
    ("Insurance & Risk Management", 15),
    ("Holistic Financial Planning", 15),
    ("Tax Planning & Optimization", 10),
    ("Estate Planning", 8),
    ("Corporate & Group Benefits", 7),
]

COMPENSATION_MODELS = [
    ("Fee-based (hybrid)", 47),
    ("Commission-based", 35),
    ("Fee-only", 18),
]

CLIENT_DEMOGRAPHICS_OPTIONS = [
    ("Mass market retail", 20),
    ("Mass affluent ($100K-$500K)", 30),
    ("High net worth ($500K-$2M)", 25),
    ("Ultra-high net worth ($2M+)", 10),
    ("Small business owners", 10),
    ("Pre-retirees and retirees", 5),
]

TECH_ADOPTION_LEVELS = [
    ("Low", 10),
    ("Medium", 30),
    ("High", 40),
    ("Very High", 20),
]

# Designations with base weights (multiple can be held)
DESIGNATIONS_POOL = {
    "CFP": 40,
    "CSC": 25,
    "CIM": 15,
    "CLU": 8,
    "PFP": 7,
    "CFA": 5,
    "RRC": 5,
    "QAFP": 4,
    "TEP": 3,
    "CHS": 2,
}

PROFESSIONAL_CHALLENGES_POOL = [
    "Client acquisition and prospecting",
    "Regulatory compliance burden",
    "Fee compression pressure",
    "Robo-advisor and fintech competition",
    "Client retention during market volatility",
    "Keeping up with product complexity",
    "Succession planning for practice",
    "Work-life balance",
    "Technology adoption costs",
    "Managing client expectations",
    "Communicating during market downturns",
    "Transitioning to fee-based model",
    "Scaling practice efficiently",
    "ESG and responsible investing demand",
    "Cybersecurity and data privacy",
    "Generational wealth transfer needs",
    "Rising client sophistication",
]

PROFESSIONAL_VALUES_POOL = [
    "Client-first fiduciary duty",
    "Long-term relationship building",
    "Continuous professional development",
    "Ethical transparency",
    "Financial literacy advocacy",
    "Innovation and technology embrace",
    "Work-life integration",
    "Community involvement and mentorship",
    "Evidence-based investing",
    "Comprehensive planning approach",
    "Fee transparency",
    "Diversity and inclusion",
    "Environmental sustainability",
    "Regulatory excellence",
]

# ============================================================
# NAME POOLS BY ETHNICITY AND GENDER
# (Same as consumer generator)
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
        "Male": ["Jean", "Pierre", "Jacques", "Michel", "Francois", "Andre", "Claude", "Marc",
                  "Louis", "Philippe", "Yves", "Alain", "Guy", "Rene", "Denis", "Bernard",
                  "Serge", "Luc", "Etienne", "Mathieu", "Gabriel", "Alexandre", "Maxime",
                  "Antoine", "Olivier", "Julien", "Nicolas", "Benoit", "Simon", "Emile"],
        "Female": ["Marie", "Isabelle", "Sophie", "Catherine", "Anne", "Julie", "Nathalie",
                    "Sylvie", "Helene", "Monique", "Celine", "Francoise", "Chantal", "Josee",
                    "Diane", "Madeleine", "Louise", "Claire", "Emilie", "Aurelie", "Camille",
                    "Lea", "Florence", "Gabrielle", "Valerie", "Genevieve", "Veronique"],
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
                    "Precious", "Grace", "Joy", "Faith", "Blessing", "Nadege", "Fatoumata",
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
    "French Canadian": ["Tremblay", "Gagnon", "Roy", "Cote", "Bouchard", "Gauthier", "Morin",
                         "Lavoie", "Fortin", "Gagne", "Ouellet", "Pelletier", "Belanger",
                         "Levesque", "Bergeron", "Leblanc", "Paquette", "Girard", "Simard",
                         "Boucher", "Caron", "Beaulieu", "Cloutier", "Dube", "Poirier",
                         "Fontaine", "Thibault", "Nadeau", "Charron", "Deschenes",
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
               "Traore", "Toure", "Abdi", "Hassan", "Baptiste", "Charles", "Denis",
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
# HELPER FUNCTIONS
# ============================================================

def weighted_choice(options_weights):
    """Choose from list of (option, weight) tuples."""
    options, weights = zip(*options_weights)
    return random.choices(options, weights=weights, k=1)[0]


def get_ethnicity_for_province(province):
    """Get ethnicity distribution adjusted for province."""
    base = dict(ETHNICITY_NATIONAL)
    mods = ETHNICITY_PROVINCE_MOD.get(province, {})
    adjusted = {}
    for eth, base_w in base.items():
        adjusted[eth] = base_w * mods.get(eth, 1.0)
    items = list(adjusted.items())
    return weighted_choice([(k, v) for k, v in items])


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


def get_family_status(age):
    """Family status based on age."""
    if age < 30:
        return weighted_choice([("Single", 55), ("Common-law partner", 20), ("Married", 20), ("Divorced", 5)])
    elif age < 40:
        return weighted_choice([("Single", 20), ("Common-law partner", 15), ("Married", 55), ("Divorced", 10)])
    elif age < 55:
        return weighted_choice([("Single", 10), ("Common-law partner", 8), ("Married", 58), ("Divorced", 18), ("Separated", 6)])
    elif age < 65:
        return weighted_choice([("Single", 8), ("Married", 55), ("Divorced", 20), ("Widowed", 8), ("Common-law partner", 9)])
    else:
        return weighted_choice([("Married", 48), ("Widowed", 18), ("Divorced", 18), ("Single", 10), ("Common-law partner", 6)])


def get_dependents(age, family_status):
    """Number of dependents."""
    if family_status == "Single" and age < 30:
        return weighted_choice([(0, 90), (1, 8), (2, 2)])
    elif age < 30:
        return weighted_choice([(0, 60), (1, 25), (2, 12), (3, 3)])
    elif age < 40:
        return weighted_choice([(0, 25), (1, 30), (2, 30), (3, 12), (4, 3)])
    elif age < 55:
        return weighted_choice([(0, 20), (1, 22), (2, 35), (3, 18), (4, 5)])
    elif age < 65:
        return weighted_choice([(0, 50), (1, 25), (2, 18), (3, 5), (4, 2)])
    else:
        return weighted_choice([(0, 80), (1, 12), (2, 6), (3, 2)])


def get_urban_rural(city):
    """Classify as urban, suburban, or rural. Advisors are mostly urban/suburban."""
    major_cities = {"Toronto", "Montreal", "Vancouver", "Calgary", "Edmonton", "Ottawa",
                    "Winnipeg", "Quebec City", "Hamilton", "Halifax", "Victoria"}
    if city in major_cities:
        return weighted_choice([("Urban", 60), ("Suburban", 38), ("Rural", 2)])
    else:
        return weighted_choice([("Urban", 35), ("Suburban", 45), ("Rural", 20)])


def get_years_in_business(age):
    """Years in business, correlated with age. Minimum career start ~24."""
    max_possible = max(0, age - 24)
    if max_possible <= 0:
        return 0
    if max_possible <= 3:
        return random.randint(0, max_possible)
    # Most advisors enter mid-to-late 20s; some later career changers
    career_start_offset = random.randint(0, min(8, max_possible // 2))
    years = max_possible - career_start_offset
    return max(0, min(years, max_possible))


def get_designations(years_in_business, education, practice_focus):
    """Get professional designations based on experience and focus."""
    # Number of designations correlated with years
    if years_in_business <= 3:
        num_desig = weighted_choice([(1, 70), (2, 25), (3, 5)])
    elif years_in_business <= 7:
        num_desig = weighted_choice([(1, 30), (2, 45), (3, 20), (4, 5)])
    elif years_in_business <= 15:
        num_desig = weighted_choice([(1, 10), (2, 35), (3, 40), (4, 15)])
    else:
        num_desig = weighted_choice([(1, 5), (2, 25), (3, 40), (4, 30)])

    # Build weighted pool with practice-focus adjustments
    pool = dict(DESIGNATIONS_POOL)
    if practice_focus in ("Insurance & Risk Management",):
        pool["CLU"] = pool.get("CLU", 8) * 2.5
        pool["CHS"] = pool.get("CHS", 2) * 2.0
    if practice_focus in ("Wealth Management", "Tax Planning & Optimization"):
        pool["CFA"] = pool.get("CFA", 5) * 2.0
        pool["CIM"] = pool.get("CIM", 15) * 1.5
    if practice_focus in ("Estate Planning",):
        pool["TEP"] = pool.get("TEP", 3) * 3.0
    if practice_focus in ("Retirement Planning", "Holistic Financial Planning"):
        pool["CFP"] = pool.get("CFP", 40) * 1.3
        pool["RRC"] = pool.get("RRC", 5) * 2.0

    # Junior advisors are more likely to have QAFP or CSC; less likely CFA/CFP
    if years_in_business <= 3:
        pool["QAFP"] = pool.get("QAFP", 4) * 3.0
        pool["CSC"] = pool.get("CSC", 25) * 1.5
        pool["CFP"] = pool.get("CFP", 40) * 0.4
        pool["CFA"] = pool.get("CFA", 5) * 0.3

    # Advanced degrees boost CFA odds
    if education in ("Master's degree", "Doctorate or professional degree"):
        pool["CFA"] = pool.get("CFA", 5) * 1.8

    # Select designations without replacement
    items = list(pool.items())
    selected = []
    for _ in range(min(num_desig, len(items))):
        choice = weighted_choice(items)
        if choice not in selected:
            selected.append(choice)
            items = [(k, w) for k, w in items if k != choice]
        if not items:
            break

    # Everyone should have at least CSC or equivalent
    if not selected:
        selected = ["CSC"]

    return selected


def get_firm_type(province):
    """Get firm type with slight provincial variation."""
    base = list(FIRM_TYPES)
    # Quebec has more insurance-based firms
    if province == "Quebec":
        base = [(ft, w * 1.4 if ft == "Insurance-based firm" else w * 0.95) for ft, w in base]
    # Alberta has more independent dealers
    elif province == "Alberta":
        base = [(ft, w * 1.3 if ft == "Independent dealer" else w) for ft, w in base]
    return weighted_choice(base)


def get_book_size_aum(years_in_business, firm_type):
    """Book size (AUM) in dollars, correlated with years and firm type."""
    # Base ranges by experience
    if years_in_business <= 3:
        base_low, base_high = 2_000_000, 25_000_000
    elif years_in_business <= 7:
        base_low, base_high = 15_000_000, 80_000_000
    elif years_in_business <= 15:
        base_low, base_high = 40_000_000, 200_000_000
    elif years_in_business <= 25:
        base_low, base_high = 80_000_000, 400_000_000
    else:
        base_low, base_high = 100_000_000, 500_000_000

    # Firm type multipliers
    firm_mult = {
        "Bank-owned brokerage": 1.3,
        "Bank retail branch": 0.6,
        "Independent dealer": 0.8,
        "Insurance-based firm": 0.7,
        "Independent RIA": 1.1,
        "Boutique / Family office": 1.8,
    }
    mult = firm_mult.get(firm_type, 1.0)

    aum = random.uniform(base_low * mult, base_high * mult)
    # Round to nearest $100K
    return max(1_000_000, round(aum / 100_000) * 100_000)


def get_num_clients(book_size_aum, client_demographics, firm_type):
    """Number of clients, inversely correlated with avg client size."""
    # Estimate avg client size from demographics
    if client_demographics in ("Ultra-high net worth ($2M+)",):
        avg_per_client = random.uniform(2_000_000, 8_000_000)
    elif client_demographics in ("High net worth ($500K-$2M)",):
        avg_per_client = random.uniform(500_000, 2_000_000)
    elif client_demographics in ("Mass affluent ($100K-$500K)",):
        avg_per_client = random.uniform(100_000, 500_000)
    elif client_demographics in ("Small business owners",):
        avg_per_client = random.uniform(200_000, 1_000_000)
    elif client_demographics in ("Pre-retirees and retirees",):
        avg_per_client = random.uniform(200_000, 800_000)
    else:  # Mass market retail
        avg_per_client = random.uniform(20_000, 150_000)

    # Bank retail tends to have more clients
    if firm_type == "Bank retail branch":
        avg_per_client *= 0.6

    clients = book_size_aum / avg_per_client
    clients *= random.uniform(0.7, 1.3)  # variation
    return max(15, min(800, round(clients)))


def get_client_demographics(firm_type, book_size_aum):
    """Client demographics based on firm type and book size."""
    if firm_type == "Boutique / Family office":
        return weighted_choice([
            ("Ultra-high net worth ($2M+)", 50), ("High net worth ($500K-$2M)", 35),
            ("Mass affluent ($100K-$500K)", 10), ("Small business owners", 5),
        ])
    elif firm_type == "Bank retail branch":
        return weighted_choice([
            ("Mass market retail", 45), ("Mass affluent ($100K-$500K)", 30),
            ("Pre-retirees and retirees", 10), ("Small business owners", 10),
            ("High net worth ($500K-$2M)", 5),
        ])
    elif firm_type == "Insurance-based firm":
        return weighted_choice([
            ("Mass market retail", 20), ("Mass affluent ($100K-$500K)", 30),
            ("Pre-retirees and retirees", 15), ("Small business owners", 15),
            ("High net worth ($500K-$2M)", 15), ("Ultra-high net worth ($2M+)", 5),
        ])
    elif firm_type == "Bank-owned brokerage":
        if book_size_aum > 200_000_000:
            return weighted_choice([
                ("High net worth ($500K-$2M)", 40), ("Ultra-high net worth ($2M+)", 25),
                ("Mass affluent ($100K-$500K)", 25), ("Small business owners", 10),
            ])
        else:
            return weighted_choice([
                ("Mass affluent ($100K-$500K)", 35), ("High net worth ($500K-$2M)", 25),
                ("Mass market retail", 15), ("Pre-retirees and retirees", 10),
                ("Small business owners", 10), ("Ultra-high net worth ($2M+)", 5),
            ])
    elif firm_type == "Independent RIA":
        return weighted_choice([
            ("High net worth ($500K-$2M)", 35), ("Mass affluent ($100K-$500K)", 25),
            ("Ultra-high net worth ($2M+)", 15), ("Small business owners", 15),
            ("Pre-retirees and retirees", 10),
        ])
    else:  # Independent dealer
        return weighted_choice(CLIENT_DEMOGRAPHICS_OPTIONS)


def get_compensation_model(firm_type):
    """Compensation model correlated with firm type."""
    if firm_type == "Independent RIA":
        return weighted_choice([("Fee-only", 50), ("Fee-based (hybrid)", 40), ("Commission-based", 10)])
    elif firm_type == "Insurance-based firm":
        return weighted_choice([("Commission-based", 60), ("Fee-based (hybrid)", 30), ("Fee-only", 10)])
    elif firm_type == "Bank retail branch":
        return weighted_choice([("Fee-based (hybrid)", 45), ("Commission-based", 45), ("Fee-only", 10)])
    elif firm_type == "Boutique / Family office":
        return weighted_choice([("Fee-only", 40), ("Fee-based (hybrid)", 50), ("Commission-based", 10)])
    else:
        return weighted_choice(COMPENSATION_MODELS)


def get_personal_income(years_in_business, book_size_aum, compensation_model, firm_type, province):
    """Personal income based on book, compensation model, firm type, and experience.

    Canadian advisor income typically ranges $50K-$300K with top earners above.
    Bank retail advisors earn less; independent/boutique advisors with large books earn more.
    Revenue yield on AUM decreases with larger books (fee compression).
    """
    # Revenue yield decreases with AUM size (fee compression)
    if book_size_aum < 25_000_000:
        rev_rate = random.uniform(0.008, 0.012)
    elif book_size_aum < 75_000_000:
        rev_rate = random.uniform(0.006, 0.010)
    elif book_size_aum < 150_000_000:
        rev_rate = random.uniform(0.005, 0.008)
    elif book_size_aum < 300_000_000:
        rev_rate = random.uniform(0.004, 0.007)
    else:
        rev_rate = random.uniform(0.003, 0.006)

    gross_revenue = book_size_aum * rev_rate

    # Payout ratio depends on compensation model and firm type
    if compensation_model == "Commission-based":
        payout_rate = random.uniform(0.30, 0.45)
    elif compensation_model == "Fee-only":
        payout_rate = random.uniform(0.45, 0.65)
    else:  # Fee-based hybrid
        payout_rate = random.uniform(0.35, 0.50)

    # Firm type adjustments — bank advisors get lower payout ratios
    if firm_type == "Bank retail branch":
        payout_rate *= 0.55  # Bank takes larger share
    elif firm_type == "Bank-owned brokerage":
        payout_rate *= 0.75
    elif firm_type == "Insurance-based firm":
        payout_rate *= 0.80
    elif firm_type == "Boutique / Family office":
        payout_rate *= 1.10  # Higher payout at boutiques

    income = gross_revenue * payout_rate

    # Province modifier
    province_mod = {
        "Ontario": 1.05, "Alberta": 1.08, "British Columbia": 1.03,
        "Quebec": 0.95, "Manitoba": 0.92, "Saskatchewan": 0.93,
        "Nova Scotia": 0.90, "New Brunswick": 0.88,
        "Newfoundland and Labrador": 0.90, "Prince Edward Island": 0.88,
        "Northwest Territories": 1.10, "Yukon": 1.05, "Nunavut": 1.05,
    }
    income *= province_mod.get(province, 1.0)

    # Individual variation
    income *= random.uniform(0.75, 1.20)

    return max(42000, round(income / 1000) * 1000)


def get_business_maturity(years_in_business):
    """Business maturity label from years in business."""
    if years_in_business <= 3:
        return "Building"
    elif years_in_business <= 7:
        return "Growing"
    elif years_in_business <= 15:
        return "Established"
    elif years_in_business <= 25:
        return "Peak"
    else:
        return "Transitioning"


def get_team_size(years_in_business, book_size_aum, firm_type):
    """Team size (including the advisor)."""
    if firm_type == "Bank retail branch":
        # Bank retail advisors are usually solo or with 1 assistant
        return weighted_choice([(1, 50), (2, 35), (3, 12), (4, 3)])
    elif book_size_aum > 200_000_000:
        return weighted_choice([(2, 10), (3, 25), (4, 25), (5, 20), (6, 10), (7, 5), (8, 5)])
    elif book_size_aum > 100_000_000:
        return weighted_choice([(1, 10), (2, 25), (3, 30), (4, 20), (5, 10), (6, 5)])
    elif years_in_business > 10:
        return weighted_choice([(1, 20), (2, 35), (3, 25), (4, 15), (5, 5)])
    else:
        return weighted_choice([(1, 50), (2, 30), (3, 15), (4, 5)])


def get_tech_adoption(age):
    """Tech adoption level correlated with age."""
    if age < 35:
        return weighted_choice([("Low", 3), ("Medium", 15), ("High", 45), ("Very High", 37)])
    elif age < 45:
        return weighted_choice([("Low", 5), ("Medium", 25), ("High", 45), ("Very High", 25)])
    elif age < 55:
        return weighted_choice([("Low", 10), ("Medium", 35), ("High", 38), ("Very High", 17)])
    else:
        return weighted_choice([("Low", 20), ("Medium", 40), ("High", 30), ("Very High", 10)])


def get_practice_focus(firm_type, designations):
    """Practice focus influenced by firm type and designations."""
    weights = dict(PRACTICE_FOCUS_OPTIONS)

    if firm_type == "Insurance-based firm":
        weights["Insurance & Risk Management"] = weights.get("Insurance & Risk Management", 15) * 2.5
        weights["Estate Planning"] = weights.get("Estate Planning", 8) * 1.5
    elif firm_type == "Boutique / Family office":
        weights["Wealth Management"] = weights.get("Wealth Management", 20) * 2.0
        weights["Estate Planning"] = weights.get("Estate Planning", 8) * 1.5
        weights["Tax Planning & Optimization"] = weights.get("Tax Planning & Optimization", 10) * 1.3
    elif firm_type == "Bank retail branch":
        weights["Holistic Financial Planning"] = weights.get("Holistic Financial Planning", 15) * 1.5
        weights["Retirement Planning"] = weights.get("Retirement Planning", 25) * 1.2

    if "CLU" in designations:
        weights["Insurance & Risk Management"] = weights.get("Insurance & Risk Management", 15) * 1.5
    if "TEP" in designations:
        weights["Estate Planning"] = weights.get("Estate Planning", 8) * 2.0
    if "CFA" in designations:
        weights["Wealth Management"] = weights.get("Wealth Management", 20) * 1.5
    if "RRC" in designations:
        weights["Retirement Planning"] = weights.get("Retirement Planning", 25) * 1.5

    return weighted_choice(list(weights.items()))


# ============================================================
# PERSONA GENERATION
# ============================================================

def generate_advisor_persona(persona_id):
    """Generate a single synthetic financial advisor persona."""
    # 1. Geography
    province = weighted_choice(list(PROVINCE_WEIGHTS.items()))
    city_options = CITIES_BY_PROVINCE[province]
    city = weighted_choice(city_options)
    urban_rural = get_urban_rural(city)

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

    # 5. Education (advisors are more educated)
    education = weighted_choice(EDUCATION_LEVELS)

    # 6. Family
    family_status = get_family_status(age)
    num_dependents = get_dependents(age, family_status)

    # 7. Professional profile
    years_in_business = get_years_in_business(age)
    firm_type = get_firm_type(province)

    # Practice focus needs designations, but designations need practice focus...
    # Resolve by picking initial practice focus, then designations, then adjusting
    initial_focus = weighted_choice(list(PRACTICE_FOCUS_OPTIONS))
    designations = get_designations(years_in_business, education, initial_focus)
    practice_focus = get_practice_focus(firm_type, designations)

    book_size_aum = get_book_size_aum(years_in_business, firm_type)
    client_demographics = get_client_demographics(firm_type, book_size_aum)
    num_clients = get_num_clients(book_size_aum, client_demographics, firm_type)
    compensation_model = get_compensation_model(firm_type)
    personal_income = get_personal_income(years_in_business, book_size_aum, compensation_model, firm_type, province)
    business_maturity = get_business_maturity(years_in_business)
    team_size = get_team_size(years_in_business, book_size_aum, firm_type)
    tech_adoption = get_tech_adoption(age)

    # Professional challenges (pick 2-3)
    num_challenges = random.randint(2, 3)
    professional_challenges = random.sample(PROFESSIONAL_CHALLENGES_POOL, num_challenges)

    # Professional values (pick 3)
    professional_values = random.sample(PROFESSIONAL_VALUES_POOL, 3)

    # 8. Build narrative summary
    pronoun_subj = "He" if gender == "Male" else ("She" if gender == "Female" else "They")
    pronoun_poss = "his" if gender == "Male" else ("her" if gender == "Female" else "their")
    verb_s = "s" if gender != "Non-binary" else ""
    verb_is = "is" if gender != "Non-binary" else "are"
    verb_has = "has" if gender != "Non-binary" else "have"

    desig_str = ", ".join(designations)

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

    # AUM formatting
    if book_size_aum >= 1_000_000_000:
        aum_str = f"${book_size_aum / 1_000_000_000:.1f}B"
    else:
        aum_str = f"${book_size_aum / 1_000_000:.0f}M"

    team_desc = f"solo practitioner" if team_size == 1 else f"leads a team of {team_size}"

    narrative = (
        f"{first_name} {last_name} is a {age}-year-old {ethnicity} {gender.lower()} "
        f"financial advisor based in {city}, {province}. "
        f"{pronoun_subj} hold{verb_s} a {education.lower()} and speak{verb_s} {primary_language.lower()}. "
        f"{pronoun_subj} {verb_has} earned {desig_str} designation{'s' if len(designations) > 1 else ''}. "
        f"{pronoun_subj} {verb_has} been in the industry for {years_in_business} year{'s' if years_in_business != 1 else ''} "
        f"and run{verb_s} a {business_maturity.lower()} practice at a {firm_type.lower()}. "
        f"{pronoun_poss.capitalize()} book of business is approximately {aum_str} across {num_clients} clients, "
        f"primarily serving {client_demographics.lower().split(' (')[0]} clients. "
        f"{pronoun_subj} {verb_is} a {team_desc} and use{verb_s} a {compensation_model.lower()} compensation model "
        f"with an approximate personal income of ${personal_income:,}/year. "
        f"{fam_desc}. "
        f"{pronoun_poss.capitalize()} practice focus is {practice_focus.lower()}. "
        f"Top professional challenges: {', '.join(professional_challenges).lower()}. "
        f"Professional values: {', '.join(professional_values).lower()}. "
        f"Tech adoption: {tech_adoption.lower()}."
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
        "family_status": family_status,
        "num_dependents": num_dependents,
        "designations": designations,
        "firm_type": firm_type,
        "years_in_business": years_in_business,
        "book_size_aum": book_size_aum,
        "num_clients": num_clients,
        "practice_focus": practice_focus,
        "compensation_model": compensation_model,
        "personal_income": personal_income,
        "business_maturity": business_maturity,
        "client_demographics": client_demographics,
        "team_size": team_size,
        "tech_adoption": tech_adoption,
        "professional_challenges": professional_challenges,
        "professional_values": professional_values,
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
        "family_status", "num_dependents", "designations", "firm_type",
        "years_in_business", "book_size_aum", "num_clients", "practice_focus",
        "compensation_model", "personal_income", "business_maturity",
        "client_demographics", "team_size", "tech_adoption",
        "professional_challenges", "professional_values", "persona_summary",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat_fields)
        writer.writeheader()
        for p in personas:
            row = dict(p)
            # Flatten lists to semicolon-separated strings
            for key in ("designations", "professional_challenges", "professional_values"):
                if isinstance(row[key], list):
                    row[key] = "; ".join(row[key])
            writer.writerow(row)


def write_json(personas, filepath):
    """Write personas to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)


def print_summary(personas):
    """Print demographic summary for verification."""

    print(f"\n{'='*60}")
    print(f"  FINANCIAL ADVISOR PERSONA GENERATION SUMMARY")
    print(f"  Total personas: {len(personas)}")
    print(f"{'='*60}")

    def show_dist(title, key):
        counts = Counter(p[key] for p in personas)
        total = len(personas)
        print(f"\n{title}:")
        for item, count in sorted(counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = "#" * int(pct / 2)
            print(f"  {item:45s} {count:4d} ({pct:5.1f}%) {bar}")

    show_dist("PROVINCE", "province")
    show_dist("GENDER", "gender")
    show_dist("ETHNICITY", "ethnicity")
    show_dist("EDUCATION", "education")
    show_dist("FIRM TYPE", "firm_type")
    show_dist("PRACTICE FOCUS", "practice_focus")
    show_dist("COMPENSATION MODEL", "compensation_model")
    show_dist("BUSINESS MATURITY", "business_maturity")
    show_dist("CLIENT DEMOGRAPHICS", "client_demographics")
    show_dist("TECH ADOPTION", "tech_adoption")
    show_dist("FAMILY STATUS", "family_status")

    # Designation frequency (multi-select)
    print(f"\nDESIGNATIONS (advisors can hold multiple):")
    desig_counts = Counter()
    for p in personas:
        for d in p["designations"]:
            desig_counts[d] += 1
    for desig, count in desig_counts.most_common():
        pct = count / len(personas) * 100
        bar = "#" * int(pct / 2)
        print(f"  {desig:10s} {count:4d} ({pct:5.1f}%) {bar}")

    # Age distribution
    print(f"\nAGE DISTRIBUTION:")
    age_brackets = [(24, 29), (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59), (60, 65), (66, 72)]
    for lo, hi in age_brackets:
        count = sum(1 for p in personas if lo <= p["age"] <= hi)
        pct = count / len(personas) * 100
        bar = "#" * int(pct / 2)
        print(f"  {lo}-{hi:2d}:  {count:4d} ({pct:5.1f}%) {bar}")

    # Years in business distribution
    print(f"\nYEARS IN BUSINESS:")
    yrs_brackets = [(0, 3), (4, 7), (8, 15), (16, 25), (26, 50)]
    yrs_labels = ["0-3 (Building)", "4-7 (Growing)", "8-15 (Established)", "16-25 (Peak)", "26+ (Transitioning)"]
    for (lo, hi), label in zip(yrs_brackets, yrs_labels):
        count = sum(1 for p in personas if lo <= p["years_in_business"] <= hi)
        pct = count / len(personas) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:30s} {count:4d} ({pct:5.1f}%) {bar}")

    # Income distribution
    print(f"\nPERSONAL INCOME DISTRIBUTION:")
    brackets = [(0, 60000), (60001, 100000), (100001, 150000), (150001, 250000), (250001, 500000), (500001, 999999999)]
    labels = ["Under $60K", "$60K-$100K", "$100K-$150K", "$150K-$250K", "$250K-$500K", "$500K+"]
    for (lo, hi), label in zip(brackets, labels):
        count = sum(1 for p in personas if lo <= p["personal_income"] <= hi)
        pct = count / len(personas) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:20s} {count:4d} ({pct:5.1f}%) {bar}")

    # Book size distribution
    print(f"\nBOOK SIZE (AUM) DISTRIBUTION:")
    aum_brackets = [(0, 25e6), (25e6+1, 75e6), (75e6+1, 150e6), (150e6+1, 300e6), (300e6+1, 999e9)]
    aum_labels = ["Under $25M", "$25M-$75M", "$75M-$150M", "$150M-$300M", "$300M+"]
    for (lo, hi), label in zip(aum_brackets, aum_labels):
        count = sum(1 for p in personas if lo <= p["book_size_aum"] <= hi)
        pct = count / len(personas) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:20s} {count:4d} ({pct:5.1f}%) {bar}")

    # Client count distribution
    print(f"\nNUMBER OF CLIENTS:")
    client_brackets = [(0, 100), (101, 200), (201, 350), (351, 500), (501, 999)]
    client_labels = ["Under 100", "100-200", "200-350", "350-500", "500+"]
    for (lo, hi), label in zip(client_brackets, client_labels):
        count = sum(1 for p in personas if lo <= p["num_clients"] <= hi)
        pct = count / len(personas) * 100
        bar = "#" * int(pct / 2)
        print(f"  {label:20s} {count:4d} ({pct:5.1f}%) {bar}")

    # Key statistics
    avg_age = sum(p["age"] for p in personas) / len(personas)
    avg_income = sum(p["personal_income"] for p in personas) / len(personas)
    avg_aum = sum(p["book_size_aum"] for p in personas) / len(personas)
    avg_clients = sum(p["num_clients"] for p in personas) / len(personas)
    avg_years = sum(p["years_in_business"] for p in personas) / len(personas)
    median_income = sorted(p["personal_income"] for p in personas)[len(personas)//2]
    median_aum = sorted(p["book_size_aum"] for p in personas)[len(personas)//2]

    print(f"\nKEY STATISTICS:")
    print(f"  Average age:             {avg_age:.1f}")
    print(f"  Average years in biz:    {avg_years:.1f}")
    print(f"  Average income:          ${avg_income:,.0f}")
    print(f"  Median income:           ${median_income:,}")
    print(f"  Average AUM:             ${avg_aum:,.0f}")
    print(f"  Median AUM:              ${median_aum:,}")
    print(f"  Average clients:         {avg_clients:.0f}")
    print(f"  Avg designations/advisor: {sum(len(p['designations']) for p in personas) / len(personas):.1f}")
    print(f"  Avg team size:           {sum(p['team_size'] for p in personas) / len(personas):.1f}")


def main():
    random.seed(SEED)

    print("Generating 1,000 synthetic Canadian financial advisor personas...")
    personas = [generate_advisor_persona(i) for i in range(NUM_PERSONAS)]

    # Write outputs
    csv_path = "advisor_personas.csv"
    json_path = "advisor_personas.json"

    write_csv(personas, csv_path)
    write_json(personas, json_path)

    print(f"\nFiles written:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")

    print_summary(personas)

    # Print sample personas
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
