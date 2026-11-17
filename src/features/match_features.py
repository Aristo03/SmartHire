import re

# Canonical skill -> list of ways it appears in text (aliases)
SKILL_ALIASES = {
    # --- Programming ---
    "python": ["python"], "java": ["java"], "javascript": ["javascript"],
    "typescript": ["typescript"], "c++": ["c++", "cpp"], "c#": ["c#"],
    "php": ["php"], "ruby": ["ruby"], "scala": ["scala"], "kotlin": ["kotlin"],
    "swift": ["swift"], "golang": ["golang"], "r": ["r programming", "rstudio"],
    "matlab": ["matlab"], "shell scripting": ["shell scripting", "bash"], "vba": ["vba"],
    # --- Web ---
    "html": ["html", "html5"], "css": ["css", "css3"],
    "react": ["react", "reactjs", "react.js"], "angular": ["angular", "angularjs"],
    "node.js": ["node.js", "nodejs", "node js"], "django": ["django"], "flask": ["flask"],
    "spring boot": ["spring boot", "spring mvc"], "hibernate": ["hibernate"],
    "jquery": ["jquery"], "bootstrap": ["bootstrap"],
    "rest api": ["rest api", "rest apis", "restful"], ".net": ["asp.net", ".net", "dotnet"],
    "wordpress": ["wordpress"], "photoshop": ["photoshop"],
    "ui/ux": ["ui/ux", "ux design", "ui design"],
    # --- Data & Databases ---
    "sql": ["sql"], "mysql": ["mysql"], "postgresql": ["postgresql", "postgres"],
    "oracle": ["oracle"], "mongodb": ["mongodb"], "nosql": ["nosql"], "pl/sql": ["pl/sql", "plsql"],
    "excel": ["excel", "ms excel"], "power bi": ["power bi", "powerbi"], "tableau": ["tableau"],
    "machine learning": ["machine learning"], "deep learning": ["deep learning"],
    "nlp": ["nlp", "natural language processing"], "data analysis": ["data analysis", "data analytics"],
    "statistics": ["statistics", "statistical"], "pandas": ["pandas"], "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn"], "tensorflow": ["tensorflow"],
    "keras": ["keras"], "pytorch": ["pytorch"], "hadoop": ["hadoop"],
    "spark": ["spark", "pyspark"], "hive": ["hive"], "etl": ["etl"],
    "data warehousing": ["data warehousing", "data warehouse"], "informatica": ["informatica"],
    "sap": ["sap"], "abap": ["abap"],
    # --- Cloud & DevOps ---
    "aws": ["aws", "amazon web services"], "azure": ["azure"], "gcp": ["gcp", "google cloud"],
    "docker": ["docker"], "kubernetes": ["kubernetes"], "jenkins": ["jenkins"],
    "git": ["git", "github"], "linux": ["linux", "unix"], "ci/cd": ["ci/cd"],
    "ansible": ["ansible"], "terraform": ["terraform"],
    # --- Testing ---
    "selenium": ["selenium"], "manual testing": ["manual testing"],
    "automation testing": ["automation testing", "test automation"],
    "junit": ["junit"], "testng": ["testng"], "jira": ["jira"], "agile": ["agile", "scrum"],
    # --- Networking & Security ---
    "network security": ["network security"], "firewall": ["firewall", "firewalls"],
    "cybersecurity": ["cybersecurity", "cyber security"], "ccna": ["ccna"],
    "tcp/ip": ["tcp/ip"], "vpn": ["vpn"],
    # --- Blockchain ---
    "blockchain": ["blockchain"], "ethereum": ["ethereum"], "solidity": ["solidity"],
    "smart contracts": ["smart contract", "smart contracts"],
    # --- Business & Management ---
    "project management": ["project management"], "business analysis": ["business analysis"],
    "requirement gathering": ["requirement gathering", "requirements gathering"],
    "stakeholder management": ["stakeholder management"],
    "ms office": ["ms office", "microsoft office"], "communication": ["communication skills"],
    "leadership": ["leadership"], "team management": ["team management", "team handling"],
    "sales": ["sales"], "marketing": ["marketing"], "digital marketing": ["digital marketing"],
    "seo": ["seo"], "crm": ["crm"], "customer service": ["customer service", "customer support"],
    "negotiation": ["negotiation"], "operations management": ["operations management"],
    "supply chain": ["supply chain"], "procurement": ["procurement"],
    "budgeting": ["budgeting"], "financial analysis": ["financial analysis"],
    "accounting": ["accounting", "accounts"], "tally": ["tally"],
    # --- HR ---
    "recruitment": ["recruitment", "recruiting", "talent acquisition"], "payroll": ["payroll"],
    "employee relations": ["employee relations"], "onboarding": ["onboarding"],
    # --- Legal ---
    "litigation": ["litigation"], "legal drafting": ["drafting"],
    "legal research": ["legal research"], "compliance": ["compliance"],
    # --- Core Engineering ---
    "autocad": ["autocad", "auto cad"], "solidworks": ["solidworks"], "catia": ["catia"],
    "revit": ["revit"], "staad pro": ["staad pro", "staad.pro", "staad"],
    "plc": ["plc"], "scada": ["scada"], "quality control": ["quality control", "qc"],
    "maintenance": ["maintenance"],
    # --- Health, Fitness & Arts ---
    "nutrition": ["nutrition"], "fitness training": ["fitness", "personal training"],
    "yoga": ["yoga"], "graphic design": ["graphic design"], "illustrator": ["illustrator"],
}


def _alias_pattern(alias: str) -> str:
    """Match the alias only as a whole word (not inside another word)."""
    return r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"


# Compile once at import time -> much faster on 21k jobs
SKILL_PATTERNS = {
    skill: re.compile("|".join(_alias_pattern(a) for a in aliases))
    for skill, aliases in SKILL_ALIASES.items()
}


def extract_skills(text) -> list[str]:
    """Return the sorted list of canonical skills found in the text."""
    text = str(text).lower()
    return sorted(skill for skill, pattern in SKILL_PATTERNS.items() if pattern.search(text))