# Traffic Flow AI - Vehicle Counting Dashboard

A premium web application for counting vehicles from videos or photos and analyzing traffic conditions. Built with a FastAPI backend (YOLOv8) and a Vite + React frontend dashboard.

## Overview

The application takes a photo or video input, uses the ultralytics YOLOv8 library to spot and count vehicles (cars, motorcycles, buses, trucks), logs the incident into a SQLite database, and returns visual confirmation through a React analytical dashboard showing live statistics and a historical chart.

## Prerequisites

Make sure you have installed on your Mac:
1. **Python 3.9+** (To run the machine learning model and backend API)
2. **Node.js v18+** (To compile and run the React web interface)

If you haven't installed developer tools on macOS, run `xcode-select --install` in your terminal first, or install Homebrew and use it to install Node and Python (`brew install node python`).

---

## 1. Starting the Backend (FastAPI + YOLOv8)

Open a terminal and run the following commands:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The server will begin running on `http://localhost:8000`. The first time you upload a photo, it will automatically download the `yolov8n.pt` model weights.

## 2. Starting the Frontend (React + Vite)

Open a **new** terminal window and run:

```bash
cd frontend
npm install
npm run dev
```

The frontend will run on a local port (typically `http://localhost:5173`). Open that URL in your browser to interact with the application.

---

## Features Built
- **Object Detection API**: Frame-by-frame sampling for videos to quickly determine peak traffic.
- **SQLite Tracker**: Keeps a log of previous uploads, timestamps, file names, counts, and calculated traffic conditions.
- **Glassmorphism UI**: High-end modern UI built strictly according to best practices, with customized dynamic dashboard charts relying on `react-chartjs-2`.
