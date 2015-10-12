CREATE TABLE fact_trips (id int, duration numeric, start_date timestamp, end_date timestamp, start_station_id int, end_station_id int, entity_id int);
# Create Table 1: fact_trips
INSERT INTO fact_trips (id, duration, start_date, end_date, start_station_id, end_station_id, entity_id) VALUES (432946, 406, '2014-08-31 22:31:00', '2014-08-31 22:38:00', 28, 32, 17);
# INSERT VALUES to this Table

CREATE TABLE dim_stations (id int, station_name varchar(60), terminal_name int);
# Create Table 2: dim_stations
INSERT INTO dim_stations (id, station_name, terminal_name) VALUES (1, 'Mountain View Caltrain Station', 28);
INSERT INTO dim_stations (id, station_name, terminal_name) VALUES (2, 'Castro Street and El Camino Real', 32);

CREATE TABLE dim_entity (id int, entity_name int, entity_type varchar(60));
# Create Table 3: dim_entity
INSERT INTO dim_entity (id, entity_name, entity_type) VALUES (1, 17, 'Subscriber');

CREATE TABLE dim_entity_zip (id int, entity_id int, zip_code varchar(60));
# Create Table 4: dim_entity
INSERT INTO dim_entity_zip (id, entity_id, zip_code) VALUES (1, 17, '94040');

CREATE TABLE dim_timezone (id int, timezone varchar(10));
# Create Table 5: dim_timezone

SELECT * FROM fact_trips ft LEFT JOIN dim_stations ds1 ON ft.start_station_id = ds1.terminal_name LEFT JOIN dim_stations ds2 ON ft.end_station_id = ds2.terminal_name;
# The above is step 1 of joining!
SELECT * FROM fact_trips ft LEFT JOIN dim_stations ds1 ON ft.start_station_id = ds1.terminal_name LEFT JOIN dim_stations ds2 ON ft.end_station_id = ds2.terminal_name LEFT JOIN dim_entity de ON ft.entity_id = de.entity_name;
# The 2nd step is to join the 3rd table
SELECT * FROM fact_trips ft LEFT JOIN dim_stations ds1 ON ft.start_station_id = ds1.terminal_name LEFT JOIN dim_stations ds2 ON ft.end_station_id = ds2.terminal_name LEFT JOIN dim_entity de ON ft.entity_id = de.entity_name LEFT JOIN dim_entity_zip dez ON ft.entity_id = dez.entity_id;
# The 3rd step is to join the 4th table
SELECT ft.id 'Trip ID', ft.duration 'Duration', ft.start_date 'Start Date', ds1.station_name 'Start Station', ft.start_station_id 'Start Terminal', ft.end_date 'End Date', ds2.station_name 'End Station', ft.end_station_id 'End Terminal', de.entity_type 'Subscriber Type', dez.zip_code 'Zip Code' FROM fact_trips ft LEFT JOIN dim_stations ds1 ON ft.start_station_id = ds1.terminal_name LEFT JOIN dim_stations ds2 ON ft.end_station_id = ds2.terminal_name LEFT JOIN dim_entity de ON ft.entity_id = de.entity_name LEFT JOIN dim_entity_zip dez ON ft.entity_id = dez.entity_id;