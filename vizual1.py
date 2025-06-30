import pygame
import numpy as np
import pandas as pd
import random
import math
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import threading
import json

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
GRID_SIZE = 8  # 8x8 grid for better visualization
CELL_SIZE = 80
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)

# Traffic light states
LIGHT_COLORS = {
    'red': RED,
    'yellow': YELLOW,
    'green': GREEN
}

class Car:
    def __init__(self, x, y, direction, speed, intersection_target, car_type='normal'):
        self.x = x
        self.y = y
        self.direction = direction  # 0=North, 1=East, 2=South, 3=West
        self.speed = speed
        self.intersection_target = intersection_target
        self.car_type = car_type  # 'normal', 'emergency'
        self.color = RED if car_type == 'emergency' else random.choice([BLUE, PURPLE, CYAN, ORANGE])
        self.size = 12 if car_type == 'emergency' else 8
        self.waiting = False
        self.wait_time = 0
        
    def update(self, intersections):
        if not self.waiting:
            # Move based on direction
            if self.direction == 0:  # North
                self.y -= self.speed
            elif self.direction == 1:  # East
                self.x += self.speed
            elif self.direction == 2:  # South
                self.y += self.speed
            elif self.direction == 3:  # West
                self.x -= self.speed
        else:
            self.wait_time += 1
            
    def draw(self, screen):
        # Draw car as rectangle
        car_rect = pygame.Rect(self.x - self.size//2, self.y - self.size//2, self.size, self.size)
        pygame.draw.rect(screen, self.color, car_rect)
        
        # Draw direction indicator
        if self.direction == 0:  # North - triangle pointing up
            points = [(self.x, self.y - self.size//2 - 3), 
                     (self.x - 3, self.y - self.size//2), 
                     (self.x + 3, self.y - self.size//2)]
        elif self.direction == 1:  # East - triangle pointing right
            points = [(self.x + self.size//2 + 3, self.y), 
                     (self.x + self.size//2, self.y - 3), 
                     (self.x + self.size//2, self.y + 3)]
        elif self.direction == 2:  # South - triangle pointing down
            points = [(self.x, self.y + self.size//2 + 3), 
                     (self.x - 3, self.y + self.size//2), 
                     (self.x + 3, self.y + self.size//2)]
        else:  # West - triangle pointing left
            points = [(self.x - self.size//2 - 3, self.y), 
                     (self.x - self.size//2, self.y - 3), 
                     (self.x - self.size//2, self.y + 3)]
        
        pygame.draw.polygon(screen, WHITE, points)

class Intersection:
    def __init__(self, row, col, x, y):
        self.row = row
        self.col = col
        self.x = x
        self.y = y
        self.id = f"{chr(65 + row)}{col + 1}"
        self.light_state = 'red'
        self.light_timer = 0
        self.light_duration = 180  # frames (~3 seconds at 60 FPS)
        self.cars_waiting = []
        self.congestion_level = 0
        self.vehicle_count = 0
        self.avg_speed = 30
        self.priority_score = 0
        self.action_recommendation = "Normal_Schedule"
        
    def update(self):
        self.light_timer += 1
        
        # Count nearby cars and calculate congestion
        self.vehicle_count = len(self.cars_waiting)
        self.congestion_level = min(10, self.vehicle_count * 0.5)
        
        # Simple traffic light cycle
        if self.light_timer >= self.light_duration:
            if self.light_state == 'red':
                self.light_state = 'green'
                self.light_duration = 240  # Green longer
            elif self.light_state == 'green':
                self.light_state = 'yellow'
                self.light_duration = 60   # Yellow shorter
            else:  # yellow
                self.light_state = 'red'
                self.light_duration = 180  # Red normal
            self.light_timer = 0
            
    def draw(self, screen, font):
        # Draw intersection
        pygame.draw.circle(screen, DARK_GRAY, (self.x, self.y), 15)
        
        # Draw traffic light
        light_color = LIGHT_COLORS[self.light_state]
        pygame.draw.circle(screen, light_color, (self.x, self.y), 8)
        
        # Draw intersection ID
        text = font.render(self.id, True, WHITE)
        screen.blit(text, (self.x - 15, self.y - 35))
        
        # Draw congestion indicator
        if self.congestion_level > 0:
            congestion_color = RED if self.congestion_level > 5 else ORANGE if self.congestion_level > 2 else YELLOW
            pygame.draw.circle(screen, congestion_color, (self.x + 20, self.y - 20), int(5 + self.congestion_level))

class TrafficModel:
    def __init__(self):
        self.le_intersection = LabelEncoder()
        self.le_day = LabelEncoder()
        self.le_weather = LabelEncoder()
        self.scaler = StandardScaler()
        self.traffic_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.is_trained = False
        self.current_weather = 'Clear'
        self.current_day = 'Monday'
        self.current_hour = 8
        
    def generate_training_data(self):
        """Generate quick training data for the model"""
        intersections = [f"{chr(65 + i)}{j + 1}" for i in range(8) for j in range(8)]
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weather_types = ['Clear', 'Rain', 'Snow', 'Fog']
        
        data = []
        for _ in range(1000):  # Smaller dataset for faster training
            intersection = random.choice(intersections)
            day = random.choice(days)
            hour = random.randint(0, 23)
            weather = random.choice(weather_types)
            
            # Simple congestion simulation
            base_congestion = 2.0
            if hour in [7, 8, 9, 17, 18, 19]:  # Rush hour
                base_congestion *= random.uniform(2, 3)
            if day in ['Saturday', 'Sunday']:
                base_congestion *= random.uniform(0.5, 1.5)
            if weather != 'Clear':
                base_congestion *= random.uniform(1.2, 1.8)
                
            congestion_time = max(0.5, base_congestion + np.random.normal(0, 0.5))
            vehicle_count = max(1, int(congestion_time * random.uniform(2, 6)))
            
            # Determine action
            if congestion_time > 6:
                action = 'Open_Immediately'
            elif congestion_time > 4:
                action = 'Open_Soon'
            elif congestion_time > 2:
                action = 'Normal_Schedule'
            else:
                action = 'Delay_Opening'
                
            data.append({
                'intersection': intersection,
                'day': day,
                'hour': hour,
                'weather': weather,
                'congestion_time': congestion_time,
                'vehicle_count': vehicle_count,
                'action': action
            })
        
        return pd.DataFrame(data)
    
    def train(self):
        """Train the traffic model"""
        df = self.generate_training_data()
        
        # Encode categorical variables
        df['intersection_encoded'] = self.le_intersection.fit_transform(df['intersection'])
        df['day_encoded'] = self.le_day.fit_transform(df['day'])
        df['weather_encoded'] = self.le_weather.fit_transform(df['weather'])
        
        # Features
        X = df[['intersection_encoded', 'day_encoded', 'hour', 'weather_encoded', 
                'congestion_time', 'vehicle_count']]
        y = df['action']
        
        # Scale and train
        X_scaled = self.scaler.fit_transform(X)
        self.traffic_classifier.fit(X_scaled, y)
        self.is_trained = True
        
    def predict_action(self, intersection_id, congestion_time, vehicle_count):
        """Predict traffic light action"""
        if not self.is_trained:
            return "Normal_Schedule"
            
        try:
            # Prepare input
            input_data = np.array([[
                self.le_intersection.transform([intersection_id])[0],
                self.le_day.transform([self.current_day])[0],
                self.current_hour,
                self.le_weather.transform([self.current_weather])[0],
                congestion_time,
                vehicle_count
            ]])
            
            input_scaled = self.scaler.transform(input_data)
            prediction = self.traffic_classifier.predict(input_scaled)[0]
            return prediction
        except:
            return "Normal_Schedule"

class TrafficSimulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Smart Traffic Management System - Live Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 36)
        
        # Create grid of intersections
        self.intersections = []
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = GRID_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
                y = GRID_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2
                intersection = Intersection(row, col, x, y)
                self.intersections.append(intersection)
        
        self.cars = []
        self.model = TrafficModel()
        self.running = True
        self.paused = False
        self.simulation_speed = 1
        self.frame_count = 0
        
        # Statistics
        self.total_cars_spawned = 0
        self.total_wait_time = 0
        self.emergency_vehicles = 0
        
        # Start model training in background
        threading.Thread(target=self.model.train, daemon=True).start()
        
    def spawn_car(self):
        """Spawn a new car at random edge"""
        if len(self.cars) > 50:  # Limit cars for performance
            return
            
        # Random spawn location at edges
        edge = random.randint(0, 3)  # 0=top, 1=right, 2=bottom, 3=left
        
        if edge == 0:  # Top edge
            x = random.randint(GRID_OFFSET_X, GRID_OFFSET_X + GRID_SIZE * CELL_SIZE)
            y = GRID_OFFSET_Y
            direction = 2  # South
        elif edge == 1:  # Right edge
            x = GRID_OFFSET_X + GRID_SIZE * CELL_SIZE
            y = random.randint(GRID_OFFSET_Y, GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE)
            direction = 3  # West
        elif edge == 2:  # Bottom edge
            x = random.randint(GRID_OFFSET_X, GRID_OFFSET_X + GRID_SIZE * CELL_SIZE)
            y = GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE
            direction = 0  # North
        else:  # Left edge
            x = GRID_OFFSET_X
            y = random.randint(GRID_OFFSET_Y, GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE)
            direction = 1  # East
        
        # Determine car type
        car_type = 'emergency' if random.random() < 0.05 else 'normal'
        speed = random.uniform(1, 3) if car_type == 'normal' else random.uniform(2, 4)
        
        # Random target intersection
        target = random.choice(self.intersections)
        
        car = Car(x, y, direction, speed, target, car_type)
        self.cars.append(car)
        self.total_cars_spawned += 1
        
        if car_type == 'emergency':
            self.emergency_vehicles += 1
    
    def update_intersections(self):
        """Update all intersections with AI predictions"""
        for intersection in self.intersections:
            # Count nearby cars
            nearby_cars = []
            for car in self.cars:
                distance = math.sqrt((car.x - intersection.x)**2 + (car.y - intersection.y)**2)
                if distance < 40:  # Within intersection range
                    nearby_cars.append(car)
            
            intersection.cars_waiting = nearby_cars
            intersection.vehicle_count = len(nearby_cars)
            
            # Calculate congestion
            emergency_nearby = any(car.car_type == 'emergency' for car in nearby_cars)
            congestion_time = max(0.5, len(nearby_cars) * 0.8 + (2 if emergency_nearby else 0))
            
            # Get AI recommendation
            if self.model.is_trained:
                recommendation = self.model.predict_action(
                    intersection.id, congestion_time, intersection.vehicle_count
                )
                intersection.action_recommendation = recommendation
                
                # Adjust light timing based on AI recommendation
                if recommendation == 'Open_Immediately' and intersection.light_state == 'red':
                    intersection.light_state = 'green'
                    intersection.light_timer = 0
                    intersection.light_duration = 300  # Longer green
                elif recommendation == 'Delay_Opening' and intersection.light_state == 'green':
                    intersection.light_duration = min(intersection.light_duration, 120)  # Shorter green
            
            intersection.update()
    
    def update_cars(self):
        """Update all cars"""
        cars_to_remove = []
        
        for car in self.cars:
            # Check if car should stop at intersection
            car.waiting = False
            for intersection in self.intersections:
                distance = math.sqrt((car.x - intersection.x)**2 + (car.y - intersection.y)**2)
                if distance < 25 and intersection.light_state == 'red':
                    if car.car_type != 'emergency':  # Emergency vehicles don't stop
                        car.waiting = True
                        self.total_wait_time += 1
                    break
            
            car.update(self.intersections)
            
            # Remove cars that are off screen
            if (car.x < 0 or car.x > WINDOW_WIDTH or 
                car.y < 0 or car.y > WINDOW_HEIGHT):
                cars_to_remove.append(car)
        
        for car in cars_to_remove:
            self.cars.remove(car)
    
    def draw_ui(self):
        """Draw user interface"""
        # Background panel
        ui_rect = pygame.Rect(GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + 20, 50, 350, 800)
        pygame.draw.rect(self.screen, DARK_GRAY, ui_rect)
        pygame.draw.rect(self.screen, WHITE, ui_rect, 2)
        
        y_offset = 70
        
        # Title
        title = self.big_font.render("Traffic Control AI", True, WHITE)
        self.screen.blit(title, (ui_rect.x + 10, y_offset))
        y_offset += 50
        
        # Statistics
        stats = [
            f"Cars Active: {len(self.cars)}",
            f"Total Spawned: {self.total_cars_spawned}",
            f"Emergency Vehicles: {self.emergency_vehicles}",
            f"Avg Wait Time: {self.total_wait_time // max(1, self.total_cars_spawned):.1f}",
            f"Model Status: {'Trained' if self.model.is_trained else 'Training...'}",
            f"Weather: {self.model.current_weather}",
            f"Time: {self.model.current_hour}:00",
            f"Day: {self.model.current_day}",
        ]
        
        for stat in stats:
            text = self.font.render(stat, True, WHITE)
            self.screen.blit(text, (ui_rect.x + 10, y_offset))
            y_offset += 25
        
        y_offset += 20
        
        # Intersection details
        header = self.font.render("High Priority Intersections:", True, YELLOW)
        self.screen.blit(header, (ui_rect.x + 10, y_offset))
        y_offset += 30
        
        # Sort intersections by priority
        priority_intersections = sorted(self.intersections, 
                                      key=lambda x: x.vehicle_count, reverse=True)[:8]
        
        for intersection in priority_intersections:
            if intersection.vehicle_count > 0:
                color = RED if intersection.vehicle_count > 5 else ORANGE if intersection.vehicle_count > 2 else YELLOW
                info = f"{intersection.id}: {intersection.vehicle_count} cars"
                text = self.font.render(info, True, color)
                self.screen.blit(text, (ui_rect.x + 10, y_offset))
                
                # Show AI recommendation
                rec_text = self.font.render(f"  → {intersection.action_recommendation}", True, CYAN)
                self.screen.blit(rec_text, (ui_rect.x + 20, y_offset + 15))
                y_offset += 40
        
        # Controls
        y_offset = ui_rect.bottom - 150
        controls = [
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Reset Simulation",
            "W - Change Weather",
            "T - Change Time",
            "D - Change Day",
            "ESC - Exit"
        ]
        
        for i, control in enumerate(controls):
            color = YELLOW if i == 0 else WHITE
            text = self.font.render(control, True, color)
            self.screen.blit(text, (ui_rect.x + 10, y_offset + i * 20))
    
    def draw_grid(self):
        """Draw the road grid"""
        # Draw roads
        for i in range(GRID_SIZE + 1):
            # Vertical roads
            x = GRID_OFFSET_X + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, 
                           (x, GRID_OFFSET_Y), 
                           (x, GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE), 3)
            
            # Horizontal roads
            y = GRID_OFFSET_Y + i * CELL_SIZE
            pygame.draw.line(self.screen, GRAY, 
                           (GRID_OFFSET_X, y), 
                           (GRID_OFFSET_X + GRID_SIZE * CELL_SIZE, y), 3)
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.cars.clear()
                    self.total_cars_spawned = 0
                    self.total_wait_time = 0
                    self.emergency_vehicles = 0
                elif event.key == pygame.K_w:
                    weathers = ['Clear', 'Rain', 'Snow', 'Fog']
                    current_idx = weathers.index(self.model.current_weather)
                    self.model.current_weather = weathers[(current_idx + 1) % len(weathers)]
                elif event.key == pygame.K_t:
                    self.model.current_hour = (self.model.current_hour + 1) % 24
                elif event.key == pygame.K_d:
                    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    current_idx = days.index(self.model.current_day)
                    self.model.current_day = days[(current_idx + 1) % len(days)]
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def run(self):
        """Main simulation loop"""
        while self.running:
            self.handle_events()
            
            if not self.paused:
                self.frame_count += 1
                
                # Spawn cars periodically
                if self.frame_count % (60 // max(1, self.simulation_speed)) == 0:
                    self.spawn_car()
                
                # Update simulation
                self.update_intersections()
                self.update_cars()
            
            # Draw everything
            self.screen.fill(BLACK)
            self.draw_grid()
            
            # Draw intersections
            for intersection in self.intersections:
                intersection.draw(self.screen, self.font)
            
            # Draw cars
            for car in self.cars:
                car.draw(self.screen)
            
            # Draw UI
            self.draw_ui()
            
            # Pause indicator
            if self.paused:
                pause_text = self.big_font.render("PAUSED", True, RED)
                self.screen.blit(pause_text, (WINDOW_WIDTH // 2 - 50, 20))
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()

if __name__ == "__main__":
    print("🚦 Starting Smart Traffic Management Visualization")
    print("=" * 50)
    print("Controls:")
    print("- SPACE: Pause/Resume")
    print("- R: Reset simulation")
    print("- W: Change weather")
    print("- T: Change time of day")
    print("- D: Change day of week")
    print("- ESC: Exit")
    print("\nLaunching visualization...")
    
    try:
        simulation = TrafficSimulation()
        simulation.run()
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()