# Imports
import os
import pygame
import argparse
import numpy as np
from random import randint
from keras.utils import to_categorical
from DQN import DQNAgent
from TrainingGraph import plot_graph

# ignores unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ----------------------------------------------------------------------------------------------------------------------
# Parameters Used
def par_def():
    # parameters dictionary
    par = dict()

    # Change between for training or trained
    par['load_weights'] = True
    par['train'] = False

    par['epsilon_decay_linear'] = 1 / 75
    par['learning_rate'] = 0.0005

    # attempted layer sizing numbers
    # 100, 150, 200
    par['first_layer_size'] = 200  # neurons in the first layer
    par['second_layer_size'] = 150  # neurons in the second layer
    par['third_layer_size'] = 100  # neurons in the third layer

    # number of games, memory and batch sizing
    #     # attempted sizes
    #     # episodes 400, 1000, 2000
    #     # memory 3000, 2500, 4000
    #     # batch sizes 500, 550, 600, 650

    # reduce for the actual weights loading for an easier visualisation
    # 400-500 minimum when training
    par['episodes'] = 400
    par['memory_size'] = 2500   # switch to 4000 when training to see what may occur
    par['batch_size'] = 500

    # model for trained
    par['weights_path'] = 'weights/weights.hdf5'

    return par


# main game classes ----------------------------------------------------------------------------------------------------
# Game class
# Food object
class Food(object):
    # initialisation
    def __init__(self):
        self.x_f = 240
        self.y_f = 200

        self.image = pygame.image.load('img/food.png')

    # displaying the food
    def food_disp(self, x, y, game):
        game.dispG.blit(self.image, (x, y))
        screen_updating()

    # food coordinates
    def find_food(self, game, snake):
        x_rand = randint(20, game.game_w - 40)
        self.x_f = x_rand - x_rand % 20

        y_rand = randint(20, game.game_h - 40)
        self.y_f = y_rand - y_rand % 20

        if [self.x_f, self.y_f] not in snake.pos:
            return self.x_f, self.y_f
        else:
            self.find_food(game, snake)


# Snake class
class snake(object):
    # Initialising the snake
    def __init__(self, game):
        # food available and checking if eaten
        self.food = 1
        self.eaten = False

        # Snake image
        self.image = pygame.image.load('img/snake.png')

        # X and Y values
        x = 0.45 * game.game_w
        y = 0.5 * game.game_h

        # positioning
        self.x = x - x % 20
        self.y = y - y % 20

        # List of positions
        self.pos = []
        self.pos.append([self.x, self.y])

        # changes in x and y
        self.x_ch = 20
        self.y_ch = 0

    # Displaying the snake
    def snake_disp(self, x, y, food, game):
        self.pos[-1][0] = x
        self.pos[-1][1] = y

        # check if crashing occurs
        if game.crash == False:
            for i in range(food):
                x_temp, y_temp = self.pos[len(self.pos) - 1 - i]
                game.dispG.blit(self.image, (x_temp, y_temp))

            # update the screen
            screen_updating()
        else:
            pygame.time.wait(200)

    # Updating the pos
    def position_update(self, x, y):
        # check positioning
        if self.pos[-1][0] != x or self.pos[-1][1] != y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.pos[i][0], self.pos[i][1] = self.pos[i + 1]

            self.pos[-1][0] = x
            self.pos[-1][1] = y

    # Moving
    def move(self, move, x, y, game, food, agent):
        move_array = [self.x_ch, self.y_ch]

        # if the snake ate the apple reset eaten and add another on the map
        if self.eaten:
            self.pos.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1

        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_ch, self.y_ch

        # right horizontally
        elif np.array_equal(move, [0, 1, 0]) and self.y_ch == 0:
            move_array = [0, self.x_ch]

        # right vertically
        elif np.array_equal(move, [0, 1, 0]) and self.x_ch == 0:
            move_array = [-self.y_ch, 0]

        # left horizontally
        elif np.array_equal(move, [0, 0, 1]) and self.y_ch == 0:
            move_array = [0, -self.x_ch]

        # left vertically
        elif np.array_equal(move, [0, 0, 1]) and self.x_ch == 0:
            move_array = [self.y_ch, 0]

        # movements made
        self.x_ch, self.y_ch = move_array
        self.x = x + self.x_ch
        self.y = y + self.y_ch

        if self.x < 20 or self.x > game.game_w - 40 \
                or self.y < 20 \
                or self.y > game.game_h - 40 \
                or [self.x, self.y] in self.pos:
            game.crash = True
        consume(self, food, game)

        # update pos
        self.position_update(self.x, self.y)


class Game:
    # Initializing the game
    def __init__(self, game_w, game_h):
        pygame.display.set_caption('##### Deep Q-Learning with Snake #####')

        # Width and height
        self.game_w = game_w
        self.game_h = game_h

        # Displaying background
        self.dispG = pygame.display.set_mode((game_w, game_h + 60))
        self.bg = pygame.image.load("img/b.png")

        # Set crash as false
        self.crash = False

        # Make make snake object food object and set score
        self.snake = snake(self)
        self.food = Food()
        self.score = 0


# additional functions -------------------------------------------------------------------------------------------------

# Displaying the user interface ----------------------------------------------------------------------------------------
def gui(game, score, rec):
    font = pygame.font.SysFont('Arial', 20)
    # font_bold = pygame.font.SysFont('Arial', 20, True)

    # score
    score_s = font.render('Score: ', True, (0, 0, 0))
    score_s_num = font.render(str(score), True, (0, 0, 0))

    # high score
    h_score = font.render('Best Score: ', True, (0, 0, 0))
    h_score_num = font.render(str(rec), True, (0, 0, 0))

    # settings
    game.dispG.blit(score_s, (45, 440))
    game.dispG.blit(score_s_num, (120, 440))
    game.dispG.blit(h_score, (190, 440))
    game.dispG.blit(h_score_num, (350, 440))
    game.dispG.blit(game.bg, (10, 10))


# Display the game
def display(snake, food, game, rec):
    game.dispG.fill((255, 255, 255))
    gui(game, game.score, rec)

    snake.snake_disp(snake.pos[-1][0], snake.pos[-1][1], snake.food, game)
    food.food_disp(food.x_f, food.y_f, game)


# Update the screen
def screen_updating():
    pygame.display.update()
    # Used for mac
    pygame.event.get()


# Get the record -------------------------------------------------------------------------------------------------------
def rec_get(score, rec):
    if score >= rec:
        return score
    else:
        return rec


# Eating function ------------------------------------------------------------------------------------------------------
def consume(sn, fd, game):
    if sn.x == fd.x_f and sn.y == fd.y_f:
        fd.find_food(game, sn)
        sn.eaten = True
        game.score = game.score + 1


# Initialising the game ------------------------------------------------------------------------------------------------
def g_init(snake, game, food, agent, batch_size):
    # gets state of the agent
    init_1 = agent.get_state(game, snake, food)

    #  sets action and moves snake
    action = [1, 0, 0]
    snake.move(action, snake.x, snake.y, game, food, agent)

    # gets state of the agent
    init_2 = agent.get_state(game, snake, food)

    # sets reward and remembering for agent
    r1 = agent.set_r(snake, game.crash)
    agent.rem(init_1, action, r1, init_2, game.crash)

    # gets and replays current until the game ends
    agent.replay_curr(agent.memory, batch_size)


# Main running function ------------------------------------------------------------------------------------------------
def start(display_option, speed, par):
    pygame.init()
    # Passing parameters to the deep q learning agent
    agent = DQNAgent(par)
    weights_filepath = par['weights_path']

    # Loading weights from saved model if not training
    if par['load_weights']:
        agent.model.load_weights(weights_filepath)
        print("weights loaded")

    count = 0
    score_pl = []
    c_pl = []
    rec = 0

    # runs while there are still episodes to play
    while count < par['episodes']:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Initialize classes
        game = Game(440, 440)
        # snake
        sn = game.snake
        # food
        fd = game.food

        # Perform first move
        g_init(sn, game, fd, agent, par['batch_size'])

        if display_option:
            display(sn, fd, game, rec)

        # Implementing the learning module
        while not game.crash:
            if not par['train']:
                agent.epsilon = 0
            else:
                # Epsilon = is set to give randomness to actions
                agent.epsilon = 1 - (count * par['epsilon_decay_linear'])

            # Get old state
            old = agent.get_state(game, sn, fd)

            # Perform random actions based on agent.epsilon, or choose the action
            if randint(0, 1) < agent.epsilon:
                # final move
                f_mv = to_categorical(randint(0, 2), num_classes=3)
            else:
                # predict action based on the old state
                pred = agent.model.predict(old.reshape((1, 11)))
                f_mv = to_categorical(np.argmax(pred[0]), num_classes=3)

            # perform new move and get new state
            sn.move(f_mv, sn.x, sn.y, game, fd, agent)
            # new state
            new_s = agent.get_state(game, sn, fd)

            # set reward for the new state
            r = agent.set_r(sn, game.crash)

            if par['train']:
                # train based on new action and state
                agent.short_mem_train(old, f_mv, r, new_s, game.crash)
                # store data into long term memory
                agent.rem(old, f_mv, r, new_s, game.crash)

            # updating the record
            rec = rec_get(game.score, rec)

            # if display is being used run display for game
            if display_option:
                display(sn, fd, game, rec)
                pygame.time.wait(speed)

        # if in training mode, use the replay_curr function for training SARAS
        if par['train']:
            agent.replay_curr(agent.memory, par['batch_size'])

        # general updating
        count += 1
        print(f'Episode: {count}      Score: {game.score}')
        print('Epsilon decay:', agent.epsilon, "\n")

        # plotting
        score_pl.append(game.score)
        c_pl.append(count)

    # if training then save the weights
    if par['train']:
        agent.model.save_weights(par['weights_path'])

    # once episodes completed plot a graph of results
    plot_graph(c_pl, score_pl)


# Setting up to run ----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # font initialising for the game display
    pygame.font.init()

    # arguments passing
    pas = argparse.ArgumentParser()
    par = par_def()

    # set options to activate or deactivate the game view, and its speed
    # use these to change speed and display options
    # 50 is a regular speed 20-30 is the visual training speed
    pas.add_argument("--display", type=bool, default=True)
    pas.add_argument("--speed", type=int, default=20)

    args = pas.parse_args()

    start(args.display, args.speed, par)
