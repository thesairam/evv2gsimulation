import tkinter as tk
from tkinter import messagebox
import requests
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster

# Function to get charging stations
def get_charging_stations(location):
    geolocator = Nominatim(user_agent="ev_app")
    location = geolocator.geocode(location)
    if not location:
        messagebox.showerror("Error", "Location not found")
        return None, None, None

    lat, lon = location.latitude, location.longitude
    url = f"https://api.openchargemap.io/v3/poi/?output=json&countrycode=US&latitude={lat}&longitude={lon}&maxresults=10&compact=true&verbose=false&key=YOUR_API_KEY"
    response = requests.get(url)
    if response.status_code != 200:
        messagebox.showerror("Error", "Failed to fetch data")
        return None, None, None

    data = response.json()
    return lat, lon, data

# Function to create map
def create_map(lat, lon, stations):
    map_ev = folium.Map(location=[lat, lon], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(map_ev)

    for station in stations:
        folium.Marker(
            location=[station['AddressInfo']['Latitude'], station['AddressInfo']['Longitude']],
            popup=f"{station['AddressInfo']['Title']}\n{station['AddressInfo']['AddressLine1']}"
        ).add_to(marker_cluster)

    map_ev.save("ev_map.html")

# Function to search and display charging stations
def search_stations():
    location = location_entry.get()
    if not location:
        messagebox.showerror("Error", "Please enter a location")
        return

    lat, lon, stations = get_charging_stations(location)
    if lat and lon and stations:
        create_map(lat, lon, stations)
        messagebox.showinfo("Success", "Map created successfully. Open ev_map.html to view the map.")

# Tkinter GUI setup
root = tk.Tk()
root.title("EV Charging Station Finder")

tk.Label(root, text="Enter Location:").pack(pady=5)
location_entry = tk.Entry(root, width=50)
location_entry.pack(pady=5)

search_button = tk.Button(root, text="Search Charging Stations", command=search_stations)
search_button.pack(pady=20)

root.mainloop()
