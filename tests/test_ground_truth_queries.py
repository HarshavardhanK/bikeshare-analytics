import psycopg2
import os
from datetime import datetime

DB_PARAMS = dict(
    dbname=os.getenv('POSTGRES_DB', 'bike-share-assessment'),
    user=os.getenv('POSTGRES_USER', 'attriassessment'),
    password=os.getenv('POSTGRES_PASSWORD'),
    host=os.getenv('POSTGRES_HOST', 'agentify-assessment.postgres.database.azure.com'),
    port=int(os.getenv('POSTGRES_PORT', 5432))
)

def run_query(sql, params=None):
    with psycopg2.connect(**DB_PARAMS) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or [])
            return cur.fetchall()

def main():
    print("\n--- Example Queries and Ground Truth ---\n")

    # 1. Total km ridden by women in June 2025
    sql1 = """
        SELECT SUM(trip_distance_km)
        FROM trips
        WHERE LOWER(rider_gender) = 'female'
          AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
    """
    ans1 = run_query(sql1)[0][0]
    print(f"Q1: How many kilometres were ridden by women in June 2025?\nA1: {ans1} km\n")

    # 2. Average ride time for trips starting at Congress Avenue in June 2025
    sql2 = """
        SELECT AVG(EXTRACT(EPOCH FROM (ended_at - started_at))/60.0)
        FROM trips t
        JOIN stations s ON t.start_station_id = s.station_id
        WHERE s.station_name = 'Congress Avenue'
          AND started_at >= '2025-06-01' AND started_at < '2025-07-01';
    """
    ans2 = run_query(sql2)[0][0]
    print(f"Q2: What was the average ride time for journeys that started at Congress Avenue in June 2025?\nA2: {round(ans2,2) if ans2 else ans2} min\n")

    # 3. Most popular end station in June 2025
    sql3 = """
        SELECT s.station_name, COUNT(*) as c
        FROM trips t
        JOIN stations s ON t.end_station_id = s.station_id
        WHERE started_at >= '2025-06-01' AND started_at < '2025-07-01'
        GROUP BY s.station_name
        ORDER BY c DESC
        LIMIT 1;
    """
    ans3 = run_query(sql3)
    print(f"Q3: Which docking point saw the most arrivals in June 2025?\nA3: {ans3[0][0]} ({ans3[0][1]} arrivals)\n")

    # 4. Total km by bike model in June 2025
    sql4 = """
        SELECT b.bike_model, SUM(t.trip_distance_km)
        FROM trips t
        JOIN bikes b ON t.bike_id = b.bike_id
        WHERE started_at >= '2025-06-01' AND started_at < '2025-07-01'
        GROUP BY b.bike_model;
    """
    ans4 = run_query(sql4)
    print("Q4: How many kilometres were ridden by each bike model in June 2025?")
    for model, km in ans4:
        print(f"  {model}: {km} km")
    print()

    # 5. Rides by non-binary riders on rainy days in June 2025
    sql5 = """
        SELECT COUNT(*)
        FROM trips t
        JOIN daily_weather w ON DATE(t.started_at) = w.weather_date
        WHERE LOWER(t.rider_gender) = 'non-binary'
          AND w.precipitation_mm > 0
          AND t.started_at >= '2025-06-01' AND t.started_at < '2025-07-01';
    """
    ans5 = run_query(sql5)[0][0]
    print(f"Q5: How many rides were taken by non-binary riders on rainy days in June 2025?\nA5: {ans5}\n")

    # 6. Youngest rider's trip in June 2025
    sql6 = """
        SELECT trip_id, rider_birth_year
        FROM trips
        WHERE started_at >= '2025-06-01' AND started_at < '2025-07-01'
        ORDER BY rider_birth_year DESC
        LIMIT 1;
    """
    ans6 = run_query(sql6)[0]
    print(f"Q6: What is the trip_id and birth year of the youngest rider in June 2025?\nA6: trip_id={ans6[0]}, birth_year={ans6[1]}\n")

if __name__ == "__main__":
    main()
