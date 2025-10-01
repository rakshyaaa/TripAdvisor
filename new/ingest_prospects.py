import pyodbc
from dateutil.parser import isoparse
from chromadb import PersistentClient
from chromadb.api.types import Documents, Embeddings
from sentence_transformers import SentenceTransformer

MSSQL = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=advancementreporting.win.louisiana.edu;"
    "DATABASE=CRM_Advance;"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

SQL = """
SELECT * from [CRM_Advance].[dbo].[view_wealth_engine_prospect_scores]
"""

CHROMA_PATH = "./wealth_engine_chroma"
COLLECTION = "wealth_engine_prospects_scores"


model = SentenceTransformer("all-MiniLM-L6-v2")


class SBERTEmbeddingFn:
    def __call__(self, input: Documents) -> Embeddings:
        embs = model.encode(input, normalize_embeddings=False)
        return embs.tolist()  # gives me list of list embeddings of text

    def name(self) -> str:
        return "sbert-all-MiniLM-L6-v2"


# ====== CHROMA CLIENT / COLLECTION ======
embedding_fn = SBERTEmbeddingFn()
client = PersistentClient(path=CHROMA_PATH)
col = client.get_or_create_collection(
    name=COLLECTION, embedding_function=embedding_fn)

# ====== EXTRACT FROM SQL AND UPSERT ======
cn = pyodbc.connect(MSSQL)
cur = cn.cursor()
cur.execute(SQL)  # connect to sql server , run query, and execute it

rows = cur.fetchall()
print("SQL row count:", len(rows))
BATCH = 128  # batch inserting rows into chroma in chunks of 128 records insteat of one at a a time
# lists to hold batch of records before upserting; ids -> unique string IDs for each doc, docs -> text chunks, metas -> metadata dicts
ids, docs, metas = [], [], []


def none_to_str(x):
    return "" if x is None else str(x)


def to_float(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None


for (
    primary_id, person_name, netWorthRating, netWorthRange, assetRating, assetRange,
    incomeRating, incomeRange, givingCapacityRating, givingCapacityRange,
    propensityToGiveRating, propensityToGiveScore1, inclinationAffiliation,
    boardMembership, influenceRating, normalized_networth_score, normalized_asset_score,
    normalized_income_score, wealth_score, givingCapacity_score, propensity_score,
    influence_score, prospect_potential_score, donor_quadrant, capacity_band,
    lifetime_giving, lifetime_giving_label, city, state, state_desc, zip_, zip5,
    county, county_desc, nation, nation_desc, town, primary_prospect_manager_list,
    segment, overall_decile, segment_decile, WEPS, prospect_tier
) in cur:

    # --- document text for embedding (short profile) ---
    doc = (
        f"{none_to_str(person_name)} — {none_to_str(city)}, {none_to_str(state)}, {none_to_str(nation)}. "
        f"Segment: {none_to_str(segment)}. Tier: {none_to_str(prospect_tier)}. "
        f"Board membership: {none_to_str(boardMembership)}. "
        f"Donor quadrant: {none_to_str(donor_quadrant)}. Capacity band: {none_to_str(capacity_band)}. "
        f"Propensity rating: {none_to_str(propensityToGiveRating)}. "
        f"Giving capacity rating: {none_to_str(givingCapacityRating)}."
    ).strip()

    # --- metadata (typed) ---
    meta = {
        "primary_id": none_to_str(primary_id),
        "person_name": none_to_str(person_name),
        "city": none_to_str(city),
        "state": none_to_str(state),
        "state_desc": none_to_str(state_desc),
        "zip": none_to_str(zip_),
        "zip5": none_to_str(zip5),
        "county": none_to_str(county),
        "county_desc": none_to_str(county_desc),
        "nation": none_to_str(nation),
        "nation_desc": none_to_str(nation_desc),
        "town": none_to_str(town),
        "primary_prospect_manager_list": none_to_str(primary_prospect_manager_list),
        "segment": none_to_str(segment),
        "overall_decile": to_float(overall_decile),
        "segment_decile": to_float(segment_decile),
        "WEPS": to_float(WEPS),
        "prospect_tier": none_to_str(prospect_tier),
        "donor_quadrant": none_to_str(donor_quadrant),
        "capacity_band": none_to_str(capacity_band),
        "boardMembership": none_to_str(boardMembership),
        "inclinationAffiliation": none_to_str(inclinationAffiliation),

        # ratings/ranges/scores
        "netWorthRating": to_float(netWorthRating),
        "netWorthRange": none_to_str(netWorthRange),
        "assetRating": to_float(assetRating),
        "assetRange": none_to_str(assetRange),
        "incomeRating": to_float(incomeRating),
        "incomeRange": none_to_str(incomeRange),
        "givingCapacityRating": to_float(givingCapacityRating),
        "givingCapacityRange": none_to_str(givingCapacityRange),
        "propensityToGiveRating": to_float(propensityToGiveRating),
        "propensityToGiveScore1": to_float(propensityToGiveScore1),
        "influenceRating": to_float(influenceRating),
        "normalized_networth_score": to_float(normalized_networth_score),
        "normalized_asset_score": to_float(normalized_asset_score),
        "normalized_income_score": to_float(normalized_income_score),
        "wealth_score": to_float(wealth_score),
        "givingCapacity_score": to_float(givingCapacity_score),
        "propensity_score": to_float(propensity_score),
        "influence_score": to_float(influence_score),
        "prospect_potential_score": to_float(prospect_potential_score),
        "lifetime_giving": to_float(lifetime_giving),
        "lifetime_giving_label": none_to_str(lifetime_giving_label),
    }

    # --- collect for batch upsert ---
    ids.append(none_to_str(primary_id))   # single chunk; ID is the PK
    docs.append(doc if doc else none_to_str(
        person_name))  # ensure non-empty doc
    metas.append(meta)

    if len(ids) >= BATCH:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        ids, docs, metas = [], [], []

if ids:
    # column upsert means, if an id already exists, it will update the record instead of creating a duplicate
    col.upsert(ids=ids, documents=docs, metadatas=metas)


cur.close()
cn.close()
print("Ingest complete with all-MiniLM-L6-v2 embeddings.")
print("Chroma row count:", col.count())
