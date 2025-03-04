import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from io import BytesIO

# ---- Page Setup ----
st.set_page_config(page_title="Sales Analysis Dashboard", layout="wide")

# ---- Custom CSS ----
st.markdown("""
    <style>
        /* Warna Sidebar */
        .sidebar .sidebar-content {
            background-color: #1E2A38 !important;
            color: white;
        }
        
        /* Warna Background Utama */
        .reportview-container {
            background-color: #F8F9FA;
        }
        
        /* Judul & Header */
        h1, h2, h3, h4 {
            color: #2E86C1;
            font-family: 'Arial', sans-serif;
        }

        /* Card Style untuk Metrics */
        .metric-card {
            background-color: #fff;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }

        /* Warna Teks Metrics */
        .metric-card h2 {
            color: #1F618D;
        }
    </style>
    """, unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.title("📌 Menu")
menu = st.sidebar.selectbox("Select Page", ["Dashboard", "Sales Analysis", "Monthly Analysis"])


# ---- Read Data ----
data = pd.read_csv('data_pembelian.csv')

# Pastikan kolom 'Date' sudah dalam format datetime dan tangani NaT
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %I:%M:%S %p', errors='coerce')
data['Date'] = data['DateTime'].dt.date
data['Hour'] = data['DateTime'].dt.hour
data['3-Hour Interval'] = (data['Hour'] // 3) * 3
data['5-Hour Interval'] = (data['Hour'] // 5) * 5
data['Month'] = data['DateTime'].dt.strftime('%Y-%m')
data['Amount'] = data['Amount'].astype(str).str.replace(',', '', regex=True).astype(float)

# Hapus nilai NaT dalam Date untuk perhitungan lebih lanjut
data = data.dropna(subset=['Date'])

# ---- Helper Functions ----
def sales_by_hour(df):
    return df.groupby('Hour').agg({'Amount': 'sum', 'Username': 'count'}).reset_index()

def sales_by_3hour(df):
    return df.groupby('3-Hour Interval').agg({'Amount': 'sum', 'Username': 'count'}).reset_index()

def sales_by_5hour(df):
    return df.groupby('5-Hour Interval').agg({'Amount': 'sum', 'Username': 'count'}).reset_index()

# ---- Dashboard ----
if menu == "Dashboard":
    st.title("📊 Sales Analysis Dashboard🚀")

    # Sales Metrics
    total_amount = data['Amount'].sum()
    total_transactions = len(data)
    unique_users = data['Username'].nunique()

    # Hitung total penjualan hari sebelumnya untuk perbandingan
    valid_dates = data['Date'].dropna()
    if not valid_dates.empty:
        last_valid_date = max(valid_dates)
        previous_day = last_valid_date - pd.Timedelta(days=1)
        previous_sales = data[data['Date'] == previous_day]['Amount'].sum()
    else:
        previous_sales = 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Sales", f"SGD {total_amount:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", f"{total_transactions:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Unique Users", f"{unique_users:,}")
        st.markdown('</div>', unsafe_allow_html=True)


    # Daily Transactions Graph
    daily_transactions = data.groupby('Date').size().reset_index(name='Total Transactions')
    fig_trans = px.line(daily_transactions, x='Date', y='Total Transactions', title="📈 Daily Total Transactions", markers=True)
    st.plotly_chart(fig_trans)

    # Pie Chart - Sales per Agent
    agent_sales = data.groupby('Agent')['Amount'].sum().reset_index()
    fig_pie = px.pie(agent_sales, names='Agent', values='Amount', title="Sales Distribution by Agent")
    st.plotly_chart(fig_pie)

elif menu == "Monthly Analysis":
    st.title("📈 Monthly Analysis")

    # Pilih agent
    selected_agent = st.selectbox("Filter by Agent", ["All"] + data['Agent'].unique().tolist())

    # Data yang difilter
    df_filtered = data.copy()
    if selected_agent != "All":
        df_filtered = df_filtered[df_filtered['Agent'] == selected_agent]

    # Group data berdasarkan bulan
    monthly_comparison = df_filtered.groupby('Month').agg({'Amount': 'sum', 'Username': 'count'}).reset_index()
    monthly_comparison.columns = ['Month', 'Total Sales', 'Total Transactions']

    # Tampilkan Dataframe
    st.dataframe(monthly_comparison, use_container_width=True)

    # --- Grafik Perbandingan Bulan ke Bulan ---
    fig_monthly = px.bar(monthly_comparison, x='Month', y=['Total Sales', 'Total Transactions'],
                         title="📊 Monthly Sales & Transactions Comparison",
                         barmode='group', color_discrete_map={"Total Sales": "blue", "Total Transactions": "orange"})
    
    st.plotly_chart(fig_monthly)


elif menu == "Sales Analysis":
    st.title("📊 Sales Analysis")

    # Pilihan Rentang Tanggal (User Bisa Pilih Sendiri)
    date_range = st.date_input("Select Date Range", 
                               [min(data["Date"]), max(data["Date"])], 
                               min_value=min(data["Date"]), 
                               max_value=max(data["Date"]))

    # Pilihan Filter Agent
    agents = ["All"] + sorted(data['Agent'].dropna().unique().tolist())
    selected_agent = st.selectbox("Filter by Agent", agents)

    # Pastikan user memilih 2 tanggal
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = data[(data["Date"] >= start_date) & (data["Date"] <= end_date)]

        # Filter berdasarkan agent (jika dipilih)
        if selected_agent != "All":
            df_filtered = df_filtered[df_filtered['Agent'] == selected_agent]

        # ---- METRICS: Total Sales & Transactions ----
        total_sales = df_filtered['Amount'].sum()
        total_transactions = len(df_filtered)
        unique_users = df_filtered['Username'].nunique()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"SGD {total_sales:,.2f}")
        col2.metric("Total Transactions", f"{total_transactions:,}")
        col3.metric("Unique Users", f"{unique_users:,}")

        # ---- Grafik Sales Trend (Harian) ----
        trend_data = df_filtered.groupby('Date').agg({'Amount': 'sum', 'Username': 'count'}).reset_index()
        fig_trend = px.line(trend_data, x='Date', y=['Amount', 'Username'], 
                            title="📈 Sales & Transactions Trend", 
                            markers=True, 
                            color_discrete_map={"Amount": "blue", "Username": "orange"})
        st.plotly_chart(fig_trend)

        # ---- Grafik Sales Per Jam ----
        hourly_sales = df_filtered.groupby('Hour').agg({'Amount': 'sum', 'Username': 'count'}).reset_index()
        fig_hourly = px.bar(hourly_sales, x='Hour', y=['Amount', 'Username'], 
                            title="⏰ Sales & Transactions Per Hour", 
                            barmode='group', 
                            color_discrete_map={"Amount": "blue", "Username": "orange"})
        st.plotly_chart(fig_hourly)

        # ---- Grafik Sales Per 5 Jam ----
        df_filtered['5-Hour Interval'] = (df_filtered['Hour'] // 5) * 5
        sales_5hour = df_filtered.groupby('5-Hour Interval').agg({'Amount': 'sum', 'Username': 'count'}).reset_index()
        fig_5hour = px.bar(sales_5hour, x='5-Hour Interval', y=['Amount', 'Username'], 
                           title="⏰ Sales & Transactions Per 5 Hours", 
                           barmode='group', 
                           color_discrete_map={"Amount": "purple", "Username": "blue"})
        st.plotly_chart(fig_5hour)

        # --- BEST-SELLING TIME ---
        best_hour = df_filtered.groupby('Hour').agg({'Amount': 'sum', 'Username': 'count'}).reset_index()
        best_hour.columns = ['Hour', 'Total Sales', 'Total Transactions']

        # Cari jam dengan transaksi terbanyak
        best_hour_top = best_hour.loc[best_hour['Total Transactions'].idxmax()]

        # Tampilkan Best-Selling Time
        st.subheader("⏰ Best-Selling Time")
        col1, col2 = st.columns(2)
        col1.metric("Best Hour", f"{best_hour_top['Hour']}:00 - {best_hour_top['Hour']+1}:00")
        col2.metric("Transactions at Best Hour", f"{best_hour_top['Total Transactions']}")

        # Tampilkan grafik transaksi per jam
        fig_best_hour = px.bar(best_hour, x='Hour', y='Total Transactions', 
                               title="⏰ Transactions Per Hour", color='Total Transactions', 
                               color_continuous_scale='Blues')
        st.plotly_chart(fig_best_hour) 

    else:
        st.warning("Please select a valid date range.")
