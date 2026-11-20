import os
import random
import pandas as pd
import numpy as np

os.makedirs('/Users/anmolrupak/.gemini/antigravity/scratch/startup-funding-idv/data', exist_ok=True)

random.seed(42)
np.random.seed(42)

startups_pool = [
    ("Flipkart", "E-Commerce", "Online Shopping Portal", "Bengaluru", "SoftBank Vision Fund, Tiger Global Management, Accel Partners", "Private Equity", 1400000000),
    ("Paytm", "FinTech", "Mobile Wallet & Payment Gateway", "Noida", "Alibaba Group, SoftBank, Berkshire Hathaway", "Private Equity", 1000000000),
    ("Ola Cabs", "Consumer Internet", "Cab Aggregator Service", "Bengaluru", "SoftBank Group, Tencent Holdings, Matrix Partners India", "Series D", 500000000),
    ("Swiggy", "Consumer Internet", "Online Food Delivery Platform", "Bangalore", "Naspers, Tencent, Hillhouse Capital", "Series G", 1000000000),
    ("Zomato", "Consumer Internet", "Restaurant Discovery & Food Delivery", "Gurgaon", "Ant Financial, Info Edge, Temasek Holdings", "Private Equity", 210000000),
    ("Byju's", "EdTech", "Online Learning App", "Bengaluru", "General Atlantic, Chan Zuckerberg Initiative, Naspers", "Private Equity", 540000000),
    ("Razorpay", "FinTech", "Payment Gateway Solutions", "Bangalore / USA", "Sequoia Capital India, Tiger Global, Y Combinator", "Series C", 75000000),
    ("CRED", "FinTech", "Credit Card Bill Payment Platform", "Bengaluru", "Sequoia Capital India, Ribbit Capital, DST Global", "Series B", 120000000),
    ("Nykaa", "E-Commerce", "Beauty and Wellness E-commerce", "Mumbai", "Steadview Capital, TPG Growth, Fidelity Management", "Private Equity", 100000000),
    ("Meesho", "E-Commerce", "Social Commerce Reselling Platform", "Bangalore", "Naspers Ventures, Sequoia Capital India, Shunwei Capital", "Series D", 125000000),
    ("PolicyBazaar", "FinTech", "Insurance Comparison Platform", "Gurgaon / New Delhi", "SoftBank Vision Fund, Info Edge, True North", "Series F", 200000000),
    ("Unacademy", "EdTech", "Online Ed-tech & Test Prep Platform", "Bangalore", "General Atlantic, Sequoia Capital India, Blume Ventures", "Series E", 110000000),
    ("Delhivery", "Logistics", "Supply Chain & Logistics Services", "Gurugram", "SoftBank Vision Fund, Carlyle Group, Tiger Global", "Series F", 413000000),
    ("Udaan", "E-Commerce", "B2B E-commerce Marketplace", "Bengaluru", "DST Global, Lightspeed Venture Partners, Altimeter Capital", "Series D", 585000000),
    ("BigBasket", "E-Commerce", "Online Grocery Platform", "Bengaluru", "Alibaba Group, Mirae Asset, CDC Group", "Series F", 150000000),
    ("Urban Company", "Consumer Internet", "Home Services Marketplace", "Gurgaon", "Vy Capital, Tiger Global, Steadview Capital", "Series E", 75000000),
    ("Pine Labs", "FinTech", "Merchant Commerce & POS Solutions", "Noida", "Temasek, PayPal Ventures, Mastercard", "Private Equity", 125000000),
    ("CarDekho", "Consumer Internet", "Automobile Marketplace", "Jaipur", "Ping An Global Voyager Fund, Sequoia Capital India", "Series D", 70000000),
    ("Pharmeasy", "HealthTech", "Online Pharmacy & Telemedicine", "Mumbai", "Temasek, CDPQ, LGT Lightstone Aspada", "Series D", 220000000),
    ("Lenskart", "E-Commerce", "Eyewear E-tailer Platform", "New Delhi", "SoftBank Vision Fund, PremjiInvest, Kedaara Capital", "Series G", 275000000),
    ("Cars24", "Consumer Internet", "Used Car Buying & Selling Platform", "Gurgaon", "DST Global, SoftBank, Exor Seeds", "Series E", 200000000),
    ("Freshworks", "Technology", "Customer Engagement SaaS Software", "Chennai", "Sequoia Capital, Accel Partners, CapitalG", "Series H", 150000000),
    ("Postman", "Technology", "API Development and Collaboration Platform", "Bengaluru", "Insight Partners, Nexus Venture Partners, CRV", "Series C", 150000000),
    ("Khatabook", "FinTech", "Digital Ledger App for SMBs", "Bengaluru", "B Capital Group, Sequoia Capital India, Falcon Edge", "Series B", 60000000),
    ("Groww", "FinTech", "Investment Platform for Stocks & Mutual Funds", "Bangalore", "Tiger Global, Sequoia Capital India, Y Combinator", "Series C", 30000000),
    ("BharatPe", "FinTech", "Merchant QR & POS Payment Network", "Delhi", "Coatue Management, Ribbit Capital, Insight Partners", "Series C", 75000000),
    ("Spinny", "Consumer Internet", "Pre-owned Car Retailing Platform", "Gurgaon", "General Catalyst, Tiger Global, Elevation Capital", "Series C", 65000000),
    ("Eruditus", "EdTech", "Executive Education Programs Platform", "Mumbai", "Prosus Ventures, Chan Zuckerberg Initiative", "Series D", 113000000),
    ("Cure.fit", "HealthTech", "Fitness, Nutrition & Healthcare Platform", "Bengaluru", "Temasek Holdings, Accel, Chiratae Ventures", "Series D", 110000000),
    ("Bounce", "Consumer Internet", "Smart Dockless Scooter Sharing", "Bengaluru", "Accel, B Capital Group, Falcon Edge Capital", "Series D", 105000000),
    ("Rapido", "Consumer Internet", "Bike Taxi Service App", "Hyderabad", "Westbridge Capital, Nexus Venture Partners", "Series B", 52000000),
    ("InCred", "FinTech", "NBFC Lending for Personal & Business", "Mumbai", "FMO, Investcorp, Paragon Partners", "Series A", 60000000),
    ("Lead School", "EdTech", "Integrated School Learning System", "Mumbai", "Westbridge Capital, Elevar Equity", "Series C", 28000000),
    ("Moglix", "E-Commerce", "B2B Industrial Goods Marketplace", "Noida", "Tiger Global, Falcon Edge Capital, Sequoia Capital India", "Series D", 60000000),
    ("Infra.Market", "E-Commerce", "B2B Procurement for Construction Materials", "Mumbai", "Tiger Global, Nexus Venture Partners", "Series B", 20000000),
    ("Rebel Foods", "Food & Beverage", "Cloud Kitchen Network", "Pune", "Coatue, Gojek, Sequoia Capital India", "Series D", 125000000),
    ("Ather Energy", "Automotive", "Electric Smart Scooter Manufacturer", "Bengaluru", "Hero MotoCorp, Sachin Bansal, InnoVen Capital", "Series C", 51000000),
    ("Practo", "HealthTech", "Doctor Search & Healthcare Appointment Booking", "Bangalore", "Tencent, Sofina, Sequoia Capital India", "Series D", 55000000),
    ("Zetwerk", "Technology", "Custom Manufacturing Marketplace Platform", "Bengaluru", "Greenoaks Capital, Lightspeed, Sequoia Capital India", "Series C", 120000000),
    ("Vedantu", "EdTech", "Live Online Tutoring Platform", "Bangalore", "Coatue, GGV Capital, Tiger Global", "Series D", 100000000),
]

investor_pool = [
    "Sequoia Capital India", "Accel Partners", "Tiger Global Management", "SoftBank Vision Fund",
    "Blume Ventures", "Kalaari Capital", "Nexus Venture Partners", "Matrix Partners India",
    "Elevation Capital", "Lightspeed Venture Partners", "Chiratae Ventures", "Info Edge",
    "India Quotient", "Orios Venture Partners", "Kae Capital", "Anupam Mittal", "Ratan Tata",
    "Kunal Shah", "Rajan Anandan", "Undisclosed Investors", "Binny Bansal", "Sachin Bansal"
]

cities_pool = [
    "Bangalore", "Bengaluru", "Mumbai", "Bombay", "New Delhi", "Delhi", "Gurgaon", "Gurugram",
    "Noida", "Pune", "Hyderabad", "Chennai", "Ahmedabad", "Kolkata", "Jaipur", "Indore",
    "Chandigarh", "Coimbatore", "Goa", "Kochi", "Bangalore / USA", "Gurgaon / New Delhi", "Mumbai / USA"
]

sector_variants = [
    "eCommerce", "E-Commerce", "ecommerce", "ECommerce", "Fin-Tech", "FinTech", "Fintech",
    "EdTech", "Edtech", "HealthTech", "Healthcare", "Consumer Internet", "Technology",
    "Logistics", "Food & Beverage", "Automotive", "AgriTech", "SaaS", "Hospitality",
    "DeepTech", "CleanTech", "Media & Entertainment", "Gaming", "Real Estate"
]

sub_verticals = [
    "Online Retail", "B2B Marketplace", "Payment Gateway", "Lending Platform",
    "K-12 Education", "Teleconsultation", "Cloud Kitchen", "Supply Chain Optimization",
    "SaaS CRM", "Fleet Management", "Hyperlocal Delivery", "InsurTech", "Wealth Management",
    "Agritech Analytics", "AI Workflow Automation", "EV Charging Infrastructure"
]

stage_variants = [
    "Seed Funding", "Seed/ Angel Funding", "Seed / Angle Funding", "Seed / Angel", "Angel Funding", "Angel",
    "Series A", "Series B", "Series C", "Series D", "Series E", "Series F",
    "Private Equity", "PrivateEquity", "Debt Funding", "Debt", "Others"
]

rows = []
sr_no = 1

# Generate 350 realistic records
for year in range(2015, 2021):
    num_deals = random.randint(50, 75)
    for _ in range(num_deals):
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        
        # Inject realistic date formatting issues
        rand_date_flavor = random.random()
        if rand_date_flavor < 0.05:
            # dot separator: 12/05.2015
            date_str = f"{day:02d}/{month:02d}.{year}"
        elif rand_date_flavor < 0.10:
            # trailing space: "13/04/2015 "
            date_str = f"{day:02d}/{month:02d}/{year} "
        elif rand_date_flavor < 0.13:
            # missing slash: "05/072018"
            date_str = f"{day:02d}/{month:02d}{year}"
        elif rand_date_flavor < 0.15:
            # unparseable/corrupted date
            date_str = f"invalid-date-{random.randint(1, 99)}"
        else:
            date_str = f"{day:02d}/{month:02d}/{year}"
            
        base = random.choice(startups_pool)
        
        # startup name (occasionally add variation)
        startup_name = base[0]
        
        # sector
        if random.random() < 0.7:
            sector = base[1]
            if sector == "E-Commerce":
                sector = random.choice(["eCommerce", "E-Commerce", "ecommerce", "ECommerce"])
            elif sector == "FinTech":
                sector = random.choice(["Fin-Tech", "FinTech", "Fintech"])
        else:
            sector = random.choice(sector_variants)
            
        sub_sector = random.choice(sub_verticals)
        
        # city
        if random.random() < 0.6:
            city = base[3]
        else:
            city = random.choice(cities_pool)
            
        # investors
        if random.random() < 0.5:
            investors = base[4]
        else:
            inv_sample = random.sample(investor_pool, random.randint(1, 3))
            investors = ", ".join(inv_sample)
            
        # stage
        if random.random() < 0.5:
            stage = base[5]
        else:
            stage = random.choice(stage_variants)
            
        # amount
        amt_rand = random.random()
        if amt_rand < 0.12:
            amount_val = "undisclosed"
        elif amt_rand < 0.16:
            amount_val = "N/A"
        elif amt_rand < 0.18:
            amount_val = "unknown"
        elif amt_rand < 0.22:
            # messy string with \xc2\xa0, $, comma
            amt_num = int(base[6] * random.uniform(0.1, 1.5))
            amount_val = f"\\xc2\\xa0${amt_num:,}"
        elif amt_rand < 0.30:
            amt_num = int(base[6] * random.uniform(0.2, 1.2))
            amount_val = f"{amt_num:,}"
        else:
            amt_num = int(base[6] * random.uniform(0.05, 1.8))
            amount_val = f"{amt_num:,}"
            
        remarks = ""
        if random.random() < 0.3:
            remarks = "Late stage expansion / acquired earlier"
            
        rows.append({
            "Sr No": sr_no,
            "Date dd/mm/yyyy": date_str,
            "Startup Name": startup_name,
            "Industry Vertical": sector,
            "SubVertical": sub_sector,
            "City Location": city,
            "Investors Name": investors,
            "InvestmentnType": stage,
            "Amount in USD": amount_val,
            "Remarks": remarks
        })
        sr_no += 1

# Add some exact duplicates to test deduplication
rows.append(rows[10].copy())
rows.append(rows[25].copy())

df = pd.DataFrame(rows)
df.to_csv('/Users/anmolrupak/.gemini/antigravity/scratch/startup-funding-idv/data/startup_funding.csv', index=False)
print(f"Generated sample dataset with {len(df)} rows at data/startup_funding.csv")
