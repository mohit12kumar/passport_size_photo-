import json
import os

# Define standard profiles to avoid duplicate definitions
EU_PASSPORT = {
    "width_mm": 35.0,
    "height_mm": 45.0,
    "dpi": 300,
    "pixel_width": 413,
    "pixel_height": 531,
    "bg_color": "White / Light Gray",
    "bg_color_hex": "#FFFFFF",
    "head_height_ratio_min": 0.70,
    "head_height_ratio_max": 0.80,
    "eye_height_ratio_min": 0.50,
    "eye_height_ratio_max": 0.62,
    "top_margin_mm": 5.0,
    "expression": "Neutral",
    "glasses": "Not Allowed (except medical reasons)",
    "head_cover": "Allowed only for religious reasons",
    "shadows": "Not Allowed",
    "smile": "Neutral",
    "rotation_max_deg": 5
}

US_PASSPORT = {
    "width_mm": 50.8,
    "height_mm": 50.8,
    "dpi": 300,
    "pixel_width": 600,
    "pixel_height": 600,
    "bg_color": "White / Off-White",
    "bg_color_hex": "#FFFFFF",
    "head_height_ratio_min": 0.50,
    "head_height_ratio_max": 0.69,
    "eye_height_ratio_min": 0.56,
    "eye_height_ratio_max": 0.69,
    "top_margin_mm": 5.0,
    "expression": "Neutral",
    "glasses": "Not Allowed",
    "head_cover": "Allowed only for religious reasons",
    "shadows": "Not Allowed",
    "smile": "Neutral",
    "rotation_max_deg": 5
}

CA_PASSPORT = {
    "width_mm": 50.0,
    "height_mm": 70.0,
    "dpi": 300,
    "pixel_width": 590,
    "pixel_height": 826,
    "bg_color": "White / Off-White",
    "bg_color_hex": "#FFFFFF",
    "head_height_ratio_min": 0.44,
    "head_height_ratio_max": 0.61,
    "eye_height_ratio_min": 0.48,
    "eye_height_ratio_max": 0.60,
    "top_margin_mm": 10.0,
    "expression": "Neutral",
    "glasses": "Allowed if no reflection",
    "head_cover": "Allowed only for religious reasons",
    "shadows": "Not Allowed",
    "smile": "Neutral",
    "rotation_max_deg": 5
}

CN_PASSPORT = {
    "width_mm": 33.0,
    "height_mm": 48.0,
    "dpi": 300,
    "pixel_width": 390,
    "pixel_height": 567,
    "bg_color": "White / Light Blue",
    "bg_color_hex": "#FFFFFF",
    "head_height_ratio_min": 0.58,
    "head_height_ratio_max": 0.68,
    "eye_height_ratio_min": 0.52,
    "eye_height_ratio_max": 0.62,
    "top_margin_mm": 5.0,
    "expression": "Neutral",
    "glasses": "Not Allowed",
    "head_cover": "Allowed only for religious reasons",
    "shadows": "Not Allowed",
    "smile": "Neutral",
    "rotation_max_deg": 3
}

# 195 UN Countries list
COUNTRIES_LIST = [
    ("Afghanistan", "AF", "EU"), ("Albania", "AL", "EU"), ("Algeria", "DZ", "EU"), ("Andorra", "AD", "EU"),
    ("Angola", "AO", "EU"), ("Antigua and Barbuda", "AG", "US"), ("Argentina", "AR", "EU"), ("Armenia", "AM", "EU"),
    ("Australia", "AU", "EU"), ("Austria", "AT", "EU"), ("Azerbaijan", "AZ", "EU"), ("Bahamas", "BS", "US"),
    ("Bahrain", "BH", "EU"), ("Bangladesh", "BD", "EU"), ("Barbados", "BB", "US"), ("Belarus", "BY", "EU"),
    ("Belgium", "BE", "EU"), ("Belize", "BZ", "US"), ("Benin", "BJ", "EU"), ("Bhutan", "BT", "EU"),
    ("Bolivia", "BO", "EU"), ("Bosnia and Herzegovina", "BA", "EU"), ("Botswana", "BW", "EU"), ("Brazil", "BR", "EU"),
    ("Brunei", "BN", "EU"), ("Bulgaria", "BG", "EU"), ("Burkina Faso", "BF", "EU"), ("Burundi", "BI", "EU"),
    ("Cabo Verde", "CV", "EU"), ("Cambodia", "KH", "EU"), ("Cameroon", "CM", "EU"), ("Canada", "CA", "CA"),
    ("Central African Republic", "CF", "EU"), ("Chad", "TD", "EU"), ("Chile", "CL", "EU"), ("China", "CN", "CN"),
    ("Colombia", "CO", "US"), ("Comoros", "KM", "EU"), ("Congo (Congo-Brazzaville)", "CG", "EU"), ("Costa Rica", "CR", "US"),
    ("Croatia", "HR", "EU"), ("Cuba", "CU", "EU"), ("Cyprus", "CY", "EU"), ("Czechia (Czech Republic)", "CZ", "EU"),
    ("Democratic Republic of the Congo", "CD", "EU"), ("Denmark", "DK", "EU"), ("Djibouti", "DJ", "EU"), ("Dominica", "DM", "US"),
    ("Dominican Republic", "DO", "US"), ("Ecuador", "EC", "US"), ("Egypt", "EG", "EU"), ("El Salvador", "SV", "US"),
    ("Equatorial Guinea", "GQ", "EU"), ("Eritrea", "ER", "EU"), ("Estonia", "EE", "EU"), ("Eswatini", "SZ", "EU"),
    ("Ethiopia", "ET", "EU"), ("Fiji", "FJ", "EU"), ("Finland", "FI", "EU"), ("France", "FR", "EU"),
    ("Gabon", "GA", "EU"), ("Gambia", "GM", "EU"), ("Georgia", "GE", "EU"), ("Germany", "DE", "EU"),
    ("Ghana", "GH", "EU"), ("Greece", "GR", "EU"), ("Grenada", "GD", "US"), ("Guatemala", "GT", "US"),
    ("Guinea", "GN", "EU"), ("Guinea-Bissau", "GW", "EU"), ("Guyana", "GY", "US"), ("Haiti", "HT", "US"),
    ("Honduras", "HN", "US"), ("Hungary", "HU", "EU"), ("Iceland", "IS", "EU"), ("India", "IN", "US"),
    ("Indonesia", "ID", "EU"), ("Iran", "IR", "EU"), ("Iraq", "IQ", "EU"), ("Ireland", "IE", "EU"),
    ("Israel", "IL", "EU"), ("Italy", "IT", "EU"), ("Jamaica", "JM", "EU"), ("Japan", "JP", "EU"),
    ("Jordan", "JO", "EU"), ("Kazakhstan", "KZ", "EU"), ("Kenya", "KE", "EU"), ("Kiribati", "KI", "EU"),
    ("Kuwait", "KW", "EU"), ("Kyrgyzstan", "KG", "EU"), ("Laos", "LA", "EU"), ("Latvia", "LV", "EU"),
    ("Lebanon", "LB", "EU"), ("Lesotho", "LS", "EU"), ("Liberia", "LR", "EU"), ("Libya", "LY", "EU"),
    ("Liechtenstein", "LI", "EU"), ("Lithuania", "LT", "EU"), ("Luxembourg", "LU", "EU"), ("Madagascar", "MG", "EU"),
    ("Malawi", "MW", "EU"), ("Malaysia", "MY", "EU"), ("Maldives", "MV", "EU"), ("Mali", "ML", "EU"),
    ("Malta", "MT", "EU"), ("Marshall Islands", "MH", "US"), ("Mauritania", "MR", "EU"), ("Mauritius", "MU", "EU"),
    ("Mexico", "MX", "EU"), ("Micronesia", "FM", "US"), ("Moldova", "MD", "EU"), ("Monaco", "MC", "EU"),
    ("Mongolia", "MN", "EU"), ("Montenegro", "ME", "EU"), ("Morocco", "MA", "EU"), ("Mozambique", "MZ", "EU"),
    ("Myanmar (Burma)", "MM", "EU"), ("Namibia", "NA", "EU"), ("Nauru", "NR", "EU"), ("Nepal", "NP", "EU"),
    ("Netherlands", "NL", "EU"), ("New Zealand", "NZ", "EU"), ("Nicaragua", "NI", "US"), ("Niger", "NE", "EU"),
    ("Nigeria", "NG", "EU"), ("North Korea", "KP", "EU"), ("North Macedonia", "MK", "EU"), ("Norway", "NO", "EU"),
    ("Oman", "OM", "EU"), ("Pakistan", "PK", "EU"), ("Palau", "PW", "US"), ("Panama", "PA", "US"),
    ("Papua New Guinea", "PG", "EU"), ("Paraguay", "PY", "EU"), ("Peru", "PE", "EU"), ("Philippines", "PH", "EU"),
    ("Poland", "PL", "EU"), ("Portugal", "PT", "EU"), ("Qatar", "QA", "EU"), ("Romania", "RO", "EU"),
    ("Russia", "RU", "EU"), ("Rwanda", "RW", "EU"), ("Saint Kitts and Nevis", "KN", "US"), ("Saint Lucia", "LC", "US"),
    ("Saint Vincent and the Grenadines", "VC", "US"), ("Samoa", "WS", "EU"), ("San Marino", "SM", "EU"),
    ("Sao Tome and Principe", "ST", "EU"), ("Saudi Arabia", "SA", "EU"), ("Senegal", "SN", "EU"),
    ("Serbia", "RS", "EU"), ("Seychelles", "SC", "EU"), ("Sierra Leone", "SL", "EU"), ("Singapore", "SG", "EU"),
    ("Slovakia", "SK", "EU"), ("Slovenia", "SI", "EU"), ("Solomon Islands", "SB", "EU"), ("Somalia", "SO", "EU"),
    ("South Africa", "ZA", "EU"), ("South Korea", "KR", "EU"), ("South Sudan", "SS", "EU"), ("Spain", "ES", "EU"),
    ("Sri Lanka", "LK", "EU"), ("Sudan", "SD", "EU"), ("Suriname", "SR", "US"), ("Sweden", "SE", "EU"),
    ("Switzerland", "CH", "EU"), ("Syria", "SY", "EU"), ("Tajikistan", "TJ", "EU"), ("Tanzania", "TZ", "EU"),
    ("Thailand", "TH", "EU"), ("Timor-Leste", "TL", "EU"), ("Togo", "TG", "EU"), ("Tonga", "TO", "EU"),
    ("Trinidad and Tobago", "TT", "US"), ("Tunisia", "TN", "EU"), ("Turkey", "TR", "EU"), ("Turkmenistan", "TM", "EU"),
    ("Tuvalu", "TV", "EU"), ("Uganda", "UG", "EU"), ("Ukraine", "UA", "EU"), ("United Arab Emirates", "AE", "EU"),
    ("United Kingdom", "GB", "EU"), ("United States", "US", "US"), ("Uruguay", "UY", "EU"), ("Uzbekistan", "UZ", "EU"),
    ("Vanuatu", "VU", "EU"), ("Holy See", "VA", "EU"), ("Venezuela", "VE", "US"), ("Vietnam", "VN", "EU"),
    ("Yemen", "YE", "EU"), ("Zambia", "ZM", "EU"), ("Zimbabwe", "ZW", "EU")
]

def generate_database():
    database = {}

    for country_name, code2, standard in COUNTRIES_LIST:
        key = country_name.lower().replace(" (czech republic)", "").replace(" (congo-brazzaville)", "").replace(" (burma)", "").replace(" ", "_")
        
        # Determine baseline passport rules based on classification
        if standard == "US":
            passport = US_PASSPORT.copy()
        elif standard == "CA":
            passport = CA_PASSPORT.copy()
        elif standard == "CN":
            passport = CN_PASSPORT.copy()
        else:
            passport = EU_PASSPORT.copy()
            
        passport["name"] = f"{country_name} - Passport"

        # Construct general profiles derived from standards
        visa = passport.copy()
        visa["name"] = f"{country_name} - Visa"
        # Visa sizes often match US standard or local Schengen standard
        if standard in ("US", "CA"):
            visa["width_mm"] = 50.8
            visa["height_mm"] = 50.8
            visa["pixel_width"] = 600
            visa["pixel_height"] = 600
            visa["bg_color"] = "White"
            visa["bg_color_hex"] = "#FFFFFF"
        else:
            visa["width_mm"] = 35.0
            visa["height_mm"] = 45.0
            visa["pixel_width"] = 413
            visa["pixel_height"] = 531
            visa["bg_color"] = "White / Light Gray"
            visa["bg_color_hex"] = "#FFFFFF"

        id_card = passport.copy()
        id_card["name"] = f"{country_name} - National ID Card"
        id_card["width_mm"] = 35.0
        id_card["height_mm"] = 45.0
        id_card["pixel_width"] = 413
        id_card["pixel_height"] = 531

        residence_permit = passport.copy()
        residence_permit["name"] = f"{country_name} - Residence Permit"
        residence_permit["width_mm"] = 35.0
        residence_permit["height_mm"] = 45.0
        residence_permit["pixel_width"] = 413
        residence_permit["pixel_height"] = 531

        driving_license = passport.copy()
        driving_license["name"] = f"{country_name} - Driving License"
        driving_license["width_mm"] = 35.0
        driving_license["height_mm"] = 45.0
        driving_license["pixel_width"] = 413
        driving_license["pixel_height"] = 531

        # Country custom updates
        if key == "united_states":
            id_card["width_mm"] = 50.8
            id_card["height_mm"] = 50.8
            id_card["pixel_width"] = 600
            id_card["pixel_height"] = 600
        elif key == "india":
            # Indian visa is 2x2 inches (US standard)
            visa["width_mm"] = 50.8
            visa["height_mm"] = 50.8
            visa["pixel_width"] = 600
            visa["pixel_height"] = 600
            passport["bg_color"] = "White"
            passport["bg_color_hex"] = "#FFFFFF"
        elif key == "brazil":
            # Brazil standard is often 30x40 mm
            passport["width_mm"] = 30.0
            passport["height_mm"] = 40.0
            passport["pixel_width"] = 354
            passport["pixel_height"] = 472
            driving_license["width_mm"] = 30.0
            driving_license["height_mm"] = 40.0
            driving_license["pixel_width"] = 354
            driving_license["pixel_height"] = 472

        database[key] = {
            "name": country_name,
            "code": code2.lower(),
            "passport": passport,
            "visa": visa,
            "id_card": id_card,
            "residence_permit": residence_permit,
            "driving_license": driving_license
        }

    # Ensure models directory exists
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    json_path = os.path.join(models_dir, "country_rules.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2, ensure_ascii=False)

    print(f"[OK] Successfully compiled {len(database)} country rule profiles to: {json_path}")

if __name__ == "__main__":
    generate_database()
