from src.services.similarity import DrugSimilarityService

def main():
    # 1. Initialize the AI
    app_brain = DrugSimilarityService()
    app_brain.load_data() # Loads huge files once

    # 2. Ask questions repeatedly (Fast!)
    while True:
        user_input = input("\nEnter drug name (or 'q' to quit): ")
        if user_input.lower() == 'q': break
        
        print(f"Analyzing '{user_input}'...")
        
        # Chemical Search
        chem_results, err = app_brain.get_similar_drugs(user_input, method="chemical")
        if err:
            print(f"❌ Error: {err}")
            continue

        print(f"\n🧪 Top Chemical Twins for {user_input}:")
        for res in chem_results[:5]:
            print(f"  - {res['name']} ({res['score']})")

        # Network Search
        net_results, err = app_brain.get_similar_drugs(user_input, method="network")
        if net_results:
            print(f"\n🧬 Top Biological Cousins for {user_input}:")
            for res in net_results[:5]:
                print(f"  - {res['name']} ({res['score']})")

if __name__ == "__main__":
    main()