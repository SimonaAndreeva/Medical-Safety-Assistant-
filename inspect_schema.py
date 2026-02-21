from sqlalchemy import create_engine, text
from src.config import settings

def inspect():
    engine = create_engine(settings.DB_URL)
    with engine.connect() as conn:
        print("🔍 TABLES IN DATABASE:")
        # List all tables
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        tables = [row[0] for row in result]
        print(tables)
        
        print("\n🔍 COLUMNS IN 'drug_targets':")
        # Check columns in drug_targets to see if we even need a join
        if 'drug_targets' in tables:
            cols = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'drug_targets'"))
            print([row[0] for row in cols])
        else:
            print("❌ 'drug_targets' table is missing too!")

if __name__ == "__main__":
    inspect()