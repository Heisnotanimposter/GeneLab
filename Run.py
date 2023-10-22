{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPkNzzuKSxM32y4NgwybJNi"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "rHvZnxN2ro2c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 281
        },
        "outputId": "7acf759d-6396-49b6-de38-558eee92c28e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "pygame 2.5.2 (SDL 2.28.2, Python 3.10.12)\n",
            "Hello from the pygame community. https://www.pygame.org/contribute.html\n"
          ]
        },
        {
          "output_type": "error",
          "ename": "KeyboardInterrupt",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-1-a9664735701d>\u001b[0m in \u001b[0;36m<cell line: 58>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     84\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     85\u001b[0m     \u001b[0;31m# Update the display\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 86\u001b[0;31m     \u001b[0mpygame\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdisplay\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mflip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     87\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     88\u001b[0m     \u001b[0;31m# Check if any individual reached the food\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ],
      "source": [
        "import pygame\n",
        "import random\n",
        "import sys\n",
        "\n",
        "# Define maze dimensions\n",
        "WIDTH, HEIGHT = 800, 600\n",
        "CELL_SIZE = 20\n",
        "MAZE_WIDTH = WIDTH // CELL_SIZE\n",
        "MAZE_HEIGHT = HEIGHT // CELL_SIZE\n",
        "\n",
        "# Define colors\n",
        "WHITE = (255, 255, 255)\n",
        "BLACK = (0, 0, 0)\n",
        "RED = (255, 0, 0)\n",
        "GREEN = (0, 255, 0)\n",
        "\n",
        "# Initialize Pygame\n",
        "pygame.init()\n",
        "screen = pygame.display.set_mode((WIDTH, HEIGHT))\n",
        "pygame.display.set_caption(\"Genetic Algorithm Maze\")\n",
        "\n",
        "# Maze generation\n",
        "maze = [[random.choice([0, 1]) for _ in range(MAZE_WIDTH)] for _ in range(MAZE_HEIGHT)]\n",
        "\n",
        "# Define the genetic algorithm population\n",
        "population = []\n",
        "\n",
        "class Individual:\n",
        "    def __init__(self):\n",
        "        self.x = 0\n",
        "        self.y = 0\n",
        "        self.fitness = 0\n",
        "\n",
        "    def move(self, dx, dy):\n",
        "        new_x = self.x + dx\n",
        "        new_y = self.y + dy\n",
        "\n",
        "        if 0 <= new_x < MAZE_WIDTH and 0 <= new_y < MAZE_HEIGHT and maze[new_y][new_x] == 0:\n",
        "            self.x = new_x\n",
        "            self.y = new_y\n",
        "\n",
        "    def calculate_fitness(self, target_x, target_y):\n",
        "        self.fitness = 1 / ((self.x - target_x) ** 2 + (self.y - target_y) ** 2 + 1)\n",
        "\n",
        "# Initialize the population\n",
        "population_size = 100\n",
        "for _ in range(population_size):\n",
        "    individual = Individual()\n",
        "    population.append(individual)\n",
        "\n",
        "# Define the target (food) location\n",
        "food_x, food_y = MAZE_WIDTH - 1, MAZE_HEIGHT - 1\n",
        "\n",
        "# Main loop\n",
        "running = True\n",
        "generation = 0\n",
        "\n",
        "while running:\n",
        "    for event in pygame.event.get():\n",
        "        if event.type == pygame.QUIT:\n",
        "            running = False\n",
        "            break\n",
        "\n",
        "    # Update individuals\n",
        "    for individual in population:\n",
        "        # Make random moves\n",
        "        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])\n",
        "        individual.move(dx, dy)\n",
        "        individual.calculate_fitness(food_x, food_y)\n",
        "\n",
        "    # Draw the maze\n",
        "    screen.fill(WHITE)\n",
        "    for y in range(MAZE_HEIGHT):\n",
        "        for x in range(MAZE_WIDTH):\n",
        "            if maze[y][x] == 1:\n",
        "                pygame.draw.rect(screen, BLACK, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))\n",
        "\n",
        "    # Draw the food\n",
        "    pygame.draw.rect(screen, RED, (food_x * CELL_SIZE, food_y * CELL_SIZE, CELL_SIZE, CELL_SIZE))\n",
        "\n",
        "    # Draw individuals\n",
        "    for individual in population:\n",
        "        pygame.draw.circle(screen, GREEN, (individual.x * CELL_SIZE + CELL_SIZE // 2, individual.y * CELL_SIZE + CELL_SIZE // 2), 5)\n",
        "\n",
        "    # Update the display\n",
        "    pygame.display.flip()\n",
        "\n",
        "    # Check if any individual reached the food\n",
        "    for individual in population:\n",
        "        if individual.x == food_x and individual.y == food_y:\n",
        "            print(f\"Food found by an individual in generation {generation}!\")\n",
        "            running = False\n",
        "            break\n",
        "\n",
        "    generation += 1\n",
        "\n",
        "pygame.quit()\n",
        "sys.exit()\n"
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "-Qb6iEzNu7Me"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}