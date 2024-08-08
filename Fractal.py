import numpy as np
import pygame
import random
import string
import os
import pickle

class FractalImageGenerator:
    def __init__(self):
        pygame.init()  # Initialize Pygame
        self.width = pygame.display.Info().current_w  # Get current screen width
        self.height = pygame.display.Info().current_h  # Get current screen height
        self.points = []
        self.connections = {}
        self.q_table = {}
        self.actions = [
            (0.1, 0, 0, 0), (-0.1, 0, 0, 0), (0, 0.1, 0, 0), (0, -0.1, 0, 0),
            (0, 0, 0.1, 0), (0, 0, -0.1, 0), (0, 0, 0, 0.1), (0, 0, 0, -0.1)
        ]
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        self.clock = pygame.time.Clock()
        self.angle_x = 0
        self.angle_y = 0
        self.angle_z = 0
        self.alpha = 255
        self.epoch = 0
        self.generation = 0
        self.font = pygame.font.SysFont(None, 36)
        self.alphabet = string.ascii_uppercase

        self.load_state()  # Load previous state if available

    def save_state(self):
        state = {
            'points': self.points,
            'connections': self.connections,
            'q_table': self.q_table,
            'angles': (self.angle_x, self.angle_y, self.angle_z),
            'epoch': self.epoch,
            'generation': self.generation
        }
        with open('fractal_state.pkl', 'wb') as f:
            pickle.dump(state, f)
        print(f"State saved after epoch {self.epoch}, generation {self.generation}")

    def load_state(self):
        try:
            with open('fractal_state.pkl', 'rb') as f:
                state = pickle.load(f)
                self.points = state['points']
                self.connections = state['connections']
                self.q_table = state['q_table']
                self.angle_x, self.angle_y, self.angle_z = state['angles']
                self.epoch = state['epoch']
                self.generation = state['generation']
                print(f"Loaded state from epoch {self.epoch}, generation {self.generation}")
        except FileNotFoundError:
            print("No previous state file found. Starting new session.")

    def generate_fractal(self, level=10):
        self.points.clear()
        self.connections.clear()
        for point in self.generate_complex_points(level):
            self.points.append(point)
            state = self.get_state(point)
            self.q_table[state] = {action: 0 for action in self.actions}
            if point not in self.connections:
                self.connections[point] = {}

        for i, point1 in enumerate(self.points):
            for j, point2 in enumerate(self.points[i+1:]):
                distance = np.linalg.norm(np.array(point1) - np.array(point2))
                if distance < 0.5:
                    letter = random.choice(self.alphabet)
                    self.connections[point1][point2] = letter
                    self.connections[point2][point1] = letter

    def generate_complex_points(self, level):
        points = []
        for x in [0, 1]:
            for y in [0, 1]:
                for z in [0, 1]:
                    for w in [0, 1]:
                        point = (x, y, z, w)
                        points.append(point)
        return points

    def get_state(self, point):
        return tuple(np.round(point, decimals=1))

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
        if np.random.rand() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            return random.choice([action for action, q in q_values.items() if q == max_q])

    def update_q_table(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0 for action in self.actions}
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        self.q_table[state][action] = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

    def update(self):
        new_points = []
        for point in self.points:
            state = self.get_state(point)
            action = self.choose_action(state)
            next_point = tuple(np.array(point) + np.array(action))
            next_state = self.get_state(next_point)
            reward = self.compute_reward(point, next_point)
            self.update_q_table(state, action, reward, next_state)
            new_points.append(next_point)
            if next_point not in self.connections:
                self.connections[next_point] = {}
            for i, p in enumerate(self.points):
                distance = np.linalg.norm(np.array(next_point) - np.array(p))
                if distance < 0.5 and p != next_point:
                    letter = random.choice(self.alphabet)
                    if p not in self.connections:
                        self.connections[p] = {}
                    self.connections[p][next_point] = letter
                    if next_point not in self.connections:
                        self.connections[next_point] = {}
                    self.connections[next_point][p] = letter

        self.points = new_points

        self.generation += 1
        if self.generation >= 1000:
            self.epoch += 1
            self.generation = 0
            self.save_state()  # Save state at the start of each new epoch

    def compute_reward(self, point, next_point):
        distance_to_origin = np.linalg.norm(np.array(next_point))
        return -distance_to_origin

    def project_to_2d(self, point):
        # Simple projection from 4D to 2D
        rotated_point = self.rotate_3d(self.rotate_4d(point))
        x = int((rotated_point[0] + 1) * self.width / 2)
        y = int((rotated_point[1] + 1) * self.height / 2)
        return x, y

    def rotate_4d(self, point):
        w, x, y, z = point
        cos_a = np.cos(self.angle_x)
        sin_a = np.sin(self.angle_x)
        cos_b = np.cos(self.angle_y)
        sin_b = np.sin(self.angle_y)
        cos_c = np.cos(self.angle_z)
        sin_c = np.sin(self.angle_z)
        
        # 4D rotation matrix application
        x_new = cos_a * x - sin_a * y
        y_new = sin_a * x + cos_a * y
        y = y_new
        
        y_new = cos_b * y - sin_b * z
        z_new = sin_b * y + cos_b * z
        z = z_new

        z_new = cos_c * z - sin_c * w
        w_new = sin_c * z + cos_c * w
        
        return x_new, y_new, z_new, w_new

    def rotate_3d(self, point):
        x, y, z, w = point
        cos_a = np.cos(self.angle_x)
        sin_a = np.sin(self.angle_x)
        cos_b = np.cos(self.angle_y)
        sin_b = np.sin(self.angle_y)
        cos_c = np.cos(self.angle_z)
        sin_c = np.sin(self.angle_z)
        
        # 3D rotation matrix application (ignoring w)
        x_new = cos_a * x - sin_a * y
        y_new = sin_a * x + cos_a * y
        y = y_new
        
        y_new = cos_b * y - sin_b * z
        z_new = sin_b * y + cos_b * z
        z = z_new

        z_new = cos_c * z - sin_c * w
        w_new = sin_c * z + cos_c * w
        
        return x_new, y_new, z_new

    def run(self):
     self.generate_fractal()
     running = True
     while running:
         for event in pygame.event.get():
             if event.type == pygame.QUIT:
                 running = False
             elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_ESCAPE:  # Exit on ESC
                     running = False

         self.update()

         self.screen.fill((0, 0, 0))

         for point in self.points:
             x, y = self.project_to_2d(point)
             pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 3)
 
         for point1, connections in self.connections.items():
             x1, y1 = self.project_to_2d(point1)
             for point2, letter in connections.items():
                 x2, y2 = self.project_to_2d(point2)
                 pygame.draw.line(self.screen, (255, 255, 255), (x1, y1), (x2, y2))
                 mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                 text_surface = self.font.render(letter, True, (255, 255, 255))
                 self.screen.blit(text_surface, (mid_x, mid_y))

        # Render the epoch and generation text
         epoch_text = self.font.render(f'Epoch: {self.epoch}', True, (255, 255, 255))
         generation_text = self.font.render(f'Generation: {self.generation}', True, (255, 255, 255))
        
        # Blit the text onto the screen at the top left corner
         self.screen.blit(epoch_text, (10, 10))
         self.screen.blit(generation_text, (10, 40))

         pygame.display.flip()
         self.clock.tick(60)

         self.angle_x += 0.01
         self.angle_y += 0.01
         self.angle_z += 0.01

     pygame.quit()


if __name__ == "__main__":
    generator = FractalImageGenerator()
    generator.run()
