
import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from google.cloud import bigquery
import openai
import os

# Set OpenAI API key (before GPT calls)
openai.api_key = os.getenv("OPEN_AI_API_KEY")

# If not set, show error
if not openai.api_key:
    st.error("❌ OpenAI API key not found. Please set the OPEN_AI_API_KEY environment variable.")
    st.stop()
# -----------------------------
# 🔧 Configuration
# -----------------------------
PROJECT_ID = "precise-antenna-451202-b2"
DATASET_ID = "CrimeDataset"
TABLE_NAME = "CrimeData"

st.set_page_config(page_title="Boston Crime Dashboard", layout="wide")

# -----------------------------
# 🎨 Aesthetic Styles
# -----------------------------
nude_palette = ['#e3d5ca', '#d5bdaf', '#a98467', '#997b66', '#7a5c41']
st.markdown("""
    <style>
    .stApp {
        background-color: #f9f6f2;
    }
    h1, h2, h3 {
        color: #4d3f37;
    }
    .css-1d391kg {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# 🔁 Load BigQuery Data
# -----------------------------
@st.cache_data
def load_bigquery_data():
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT * 
        FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`
        WHERE Lat IS NOT NULL AND Long IS NOT NULL
    """
    df = client.query(query).to_dataframe()
    df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])
    return df

df = load_bigquery_data()

# -----------------------------
# 📋 Sidebar Filters
# -----------------------------
st.sidebar.title("🔍 Filter Crime Data")
min_date = df["OCCURRED_ON_DATE"].min().date()
max_date = df["OCCURRED_ON_DATE"].max().date()

date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
offense_types = st.sidebar.multiselect("Crime Types", df["OFFENSE_DESCRIPTION"].unique(), default=df["OFFENSE_DESCRIPTION"].unique())
districts = st.sidebar.multiselect("Districts", df["DISTRICT"].dropna().unique(), default=df["DISTRICT"].dropna().unique())

# -----------------------------
# 🔎 Filter Dataset
# -----------------------------
filtered = df[
    (df["OCCURRED_ON_DATE"].dt.date >= date_range[0]) &
    (df["OCCURRED_ON_DATE"].dt.date <= date_range[1]) &
    (df["OFFENSE_DESCRIPTION"].isin(offense_types)) &
    (df["DISTRICT"].isin(districts))
]

# -----------------------------
# 📊 Dashboard Visualizations
# -----------------------------
st.title("📊 Boston Crime Dashboard")

# -----------------------------
# 🤖 GPT-4 Generated SQL Section
# -----------------------------
st.divider()
st.header("AI Assitant")

user_question = st.text_area("Ask me any question related to boston crime", placeholder="e.g., What are the top 5 most frequent crimes in each district?")

if st.button("Run"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        client = bigquery.Client(project=PROJECT_ID)
        table_ref = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"
        schema = client.get_table(table_ref).schema
        schema_str = "\n".join([f"{field.name}: {field.field_type}" for field in schema])

       # Compose GPT prompt with the full table path
        full_table_path = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}"

        prompt = f"""
        You are a BigQuery SQL expert.

        Based on the schema below, generate a BigQuery Standard SQL query to answer the user's question.
        Always use this full table path in your FROM clause: `{full_table_path}`

        Schema:
        {schema_str}

        User Question:
        {user_question}

        Return ONLY the SQL query. No explanation. No markdown.
        """


        try:
            with st.spinner("Thinking..."):
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You write SQL queries for BigQuery datasets."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                sql = response.choices[0].message.content.strip().strip("```sql").strip("```")
                #st.code(sql, language="sql")

            # Execute SQL on BigQuery
            with st.spinner("Running query on the dataset..."):
                result_df = client.query(sql).to_dataframe()
                st.success("Query execution completed!!")
                st.dataframe(result_df)

                # # Optional chart
                # if result_df.shape[1] >= 2 and pd.api.types.is_numeric_dtype(result_df.iloc[:, 1]):
                #     st.bar_chart(result_df.set_index(result_df.columns[0]))

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# 📍 Map
st.header("🗺️ Incident Locations")
map_data = filtered.rename(columns={"Lat": "lat", "Long": "lon"})
st.map(map_data[["lat", "lon"]])

# 🔥 Heatmap
st.subheader("🔥 Crime Hotspot Heatmap")
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v10',
    initial_view_state=pdk.ViewState(
        latitude=map_data["lat"].mean(),
        longitude=map_data["lon"].mean(),
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HeatmapLayer',
            data=map_data,
            get_position='[lon, lat]',
            radius=200,
            elevation_scale=4,
        ),
    ],
))

# 📈 Time Series
st.subheader("📈 Daily Crime Trend")
daily_crime = filtered["OCCURRED_ON_DATE"].dt.date.value_counts().sort_index()
ts_df = pd.DataFrame({"Date": daily_crime.index, "Count": daily_crime.values})
fig_ts = px.line(ts_df, x="Date", y="Count", title="Crime Over Time")
st.plotly_chart(fig_ts, use_container_width=True)

# ⏰ Hourly Crime Distribution
st.subheader("⏰ Crimes by Hour")
hour_count = filtered["HOUR"].value_counts().sort_index()
fig_hour = px.bar(x=hour_count.index, y=hour_count.values,
                  labels={'x': 'Hour', 'y': 'Crime Count'},
                  color_discrete_sequence=[nude_palette[1]])
st.plotly_chart(fig_hour, use_container_width=True)

# 📆 Day of Week
st.subheader("📆 Crimes by Day of Week")
filtered["DAY_OF_WEEK"] = filtered["DAY_OF_WEEK"].str.title()
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_count = filtered["DAY_OF_WEEK"].value_counts().reindex(dow_order).fillna(0)
fig_dow = px.bar(x=dow_count.index, y=dow_count.values,
                 labels={'x': 'Day', 'y': 'Crime Count'},
                 color_discrete_sequence=[nude_palette[2]])
st.plotly_chart(fig_dow, use_container_width=True)

# 🗺️ Districts
st.subheader("🗺️ Crimes by District")
district_crime = filtered["DISTRICT"].value_counts().reset_index()
district_crime.columns = ["District", "Crime Count"]
fig_choro = px.bar(district_crime, x="District", y="Crime Count", title="Crimes by District")
st.plotly_chart(fig_choro, use_container_width=True)