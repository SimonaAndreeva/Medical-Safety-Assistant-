import pandas as pd
from sqlalchemy import create_engine
from src.models.tier_2_network.ppi_network_model import PPINetworkModel

DB_URL = "postgresql://admin:12345@127.0.0.1:5435/medical_safety_db"

def test_real_interactions():
    print("🚀 Initializing Tier II Network Model...")
    model = PPINetworkModel(db_url=DB_URL)
    engine = create_engine(DB_URL)

    print("\n📥 Fetching 5 known drug interactions with valid targets...")
    # SQL query to get interactions where BOTH drugs actually have targets in the database
    query = """
        SELECT DISTINCT di.drug_a_id, di.drug_b_id
        FROM drug_interactions di
        INNER JOIN drug_targets dt1 ON di.drug_a_id = dt1.drug_id
        INNER JOIN drug_targets dt2 ON di.drug_b_id = dt2.drug_id
        LIMIT 5;
    """
    real_interactions = pd.read_sql(query, engine)

    print("\n🧪 Calculating RWR Network Similarity (S_net):\n")
    
    for index, row in real_interactions.iterrows():
        d1 = row['drug_a_id']
        d2 = row['drug_b_id']
        
        # Run the RWR math engine
        score = model.get_network_similarity(d1, d2)
        
        # Print results with a simple visual bar
        bar_length = int(score * 20)
        bar = "█" * bar_length if bar_length > 0 else ""
        print(f"Drug {d1} & Drug {d2} | S_net Score: {score:.4f} {bar}")

    print("\n✅ Test Complete!")

if __name__ == "__main__":
    test_real_interactions()