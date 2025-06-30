# Smart Traffic Management System Visualization

This project provides a real-time traffic simulation and management visualization using Pygame, integrated with machine learning models to predict traffic congestion and optimize traffic light actions.

## Features

- Simulates an 8x8 grid of intersections with traffic lights.
- Visualizes cars moving through the grid with different types (normal and emergency).
- Traffic lights change states based on AI model predictions.
- User controls to pause/resume, reset simulation, change weather, time, and day.
- Background training of RandomForest models for traffic congestion prediction.
- Logs simulation data to Excel files stored in a `logs` folder.

## Requirements

- Python 3.7+
- Pygame
- NumPy
- Pandas
- scikit-learn
- openpyxl

## Installation

Install all required packages using pip with the following command:

```bash
pip install pygame numpy pandas scikit-learn openpyxl
```

## Usage

Run the visualization script:

```bash
python viz2.py
```

### Controls

- `SPACE`: Pause/Resume simulation
- `R`: Reset simulation
- `W`: Change weather condition (Clear, Rain, Snow, Fog)
- `T`: Change time of day (hour)
- `D`: Change day of the week
- `ESC`: Exit simulation

## Logging

Simulation data is logged every frame and saved as an Excel file in the `logs` directory upon exiting the simulation. The log includes:

- Frame number
- Number of active cars
- Total cars spawned
- Number of emergency vehicles
- Average wait time
- Model training status
- Current weather, time, and day
- Vehicle count and AI action recommendation for each intersection

## Project Structure

- `viz2.py`: Main visualization and simulation script.
- `test1.py`: Traffic management system with data generation and model training.
- `vizual1.py`: Alternative traffic simulation visualization.
- `veztest.py`: Additional test or utility script (dependencies may vary).

## License

This project is provided as-is for educational and demonstration purposes.
