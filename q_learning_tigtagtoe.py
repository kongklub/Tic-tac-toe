# -*- coding: utf-8 -*-
import numpy as np
import random
import tkinter as tk
from tkinter import messagebox
import pickle
import os

# Initialize the Q-table
qTable = {}
representStates = [0, 1, 2]

def getHashValue(hash):
    if hash not in qTable:
        qTable[hash] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    return qTable[hash]

def updateHash(hash, newValue):
    qTable[hash] = newValue

def getPossibilityActions(hash):
    return np.array([0 if int(val) != 0 else 1 for val in hash])

def stateToHash(state):
    return ''.join(map(str, map(int, state)))

# Create Agent class
class Agent:
    def __init__(self, epsilon=0.3, lr=0.3, gamma=.99, isPlay=False):
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.isPlay = isPlay

    def act(self, state):
        rand = random.uniform(0, 1)
        hash = stateToHash(state)
        possibilityActions = getPossibilityActions(hash)
        qValues = getHashValue(hash)

        if rand < self.epsilon and not self.isPlay:
            qValues = np.random.rand(9)

        qValues = np.array(qValues)
        if qValues.min() < 0:
            base = abs(qValues.min())
            qValues += base * 2
        qValues = np.multiply(qValues, possibilityActions)

        if qValues.sum() == 0:
            qValues = possibilityActions

        if np.count_nonzero(qValues == qValues.max()) > 1:
            bestActions = [i for i in range(len(qValues)) if qValues[i] == qValues.max()]
            return random.choice(bestActions)
        
        return np.argmax(qValues)

    def learn(self, state, nextState, action, reward, isDone):
        hashState = stateToHash(state)
        hashNextState = stateToHash(nextState)

        qState = getHashValue(hashState)
        qNextState = getHashValue(hashNextState)

        possibilityActions = getPossibilityActions(hashNextState)
        qNextState = np.multiply(qNextState, possibilityActions)

        tmpQNextState = np.array(qNextState, copy=True)
        if qNextState.min() < 0:
            base = abs(qNextState.min())
            tmpQNextState += base * 2

        qState[action] += self.lr * (reward + self.gamma * qNextState[np.argmax(tmpQNextState)] - qState[action])
        if isDone:
            qState[action] = reward

        updateHash(hashState, qState)

# Create Env class
class Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((9,))
        self.isXTurn = True
        return self.getState()

    def checkRows(self, board):
        for row in board:
            if len(set(row)) == 1 and row[0] != 0:
                return row[0]
        return 0

    def checkDiagonals(self, board):
        if len(set([board[i][i] for i in range(len(board))])) == 1:
            return board[0][0]
        if len(set([board[i][len(board) - i - 1] for i in range(len(board))])) == 1:
            return board[0][len(board) - 1]
        return 0

    def checkWin(self):
        board = self.board.reshape((3, 3))
        for newBoard in [board, np.transpose(board)]:
            result = self.checkRows(newBoard)
            if result:
                return result
        return self.checkDiagonals(board)

    def checkDraw(self):
        return self.checkWin() == 0 and not (self.board == 0).any()

    def checkDone(self):
        return self.board.min() != 0 or self.checkWin() != 0

    def getState(self):
        return np.array(self.board, copy=True)

    def act(self, action):
        reward = 0
        player = 1 if self.isXTurn else 2
        self.board[action] = player
        self.isXTurn = not self.isXTurn
        winner = self.checkWin()
        isDone = self.checkDone()

        if winner:
            reward = 1
        elif self.checkDraw():
            reward = 0.5

        nextState = np.array(self.board, copy=True)
        return nextState, reward, isDone, {}

# Train the Q-learning agent
env = Env()
agent = Agent(epsilon=0, isPlay=True)  # Use the trained model (no random exploration)

episodes = 50000
winner_history = []

def swapSide(state):
    return np.array([2 if val == 1 else 1 if val == 2 else 0 for val in state])

def rotate(state, n=1):
    return np.rot90(state.reshape((3, 3)), n).flatten()

def rotateAction(action, n=1):
    board = np.zeros((9,))
    board[action] = 1
    board = rotate(board, n)
    return np.argmax(board)

for episode in range(episodes):
    isDone = False
    state = env.reset()
    prevState = state
    prevAction = -1
    isShouldLearn = False

    if episode % 1000 == 0:
        print("episode:", episode)

    while not isDone:
        state = env.getState()

        if not env.isXTurn:
            state = swapSide(state)

        action = agent.act(state)
        nextState, reward, isDone, _ = env.act(action)

        if env.isXTurn:
            nextState = swapSide(nextState)

        if isShouldLearn:
            if isDone and not env.checkDraw():
                prevReward = -1
            elif isDone and env.checkDraw():
                prevReward = 0.5
            agent.learn(prevState, swapSide(nextState), prevAction, prevReward, isDone)
            for rotation in range(1, 4):
                agent.learn(rotate(prevState, rotation), rotate(swapSide(nextState), rotation), rotateAction(prevAction, rotation), prevReward, isDone)

        if isDone:
            agent.learn(state, nextState, action, reward, isDone)
            for rotation in range(1, 4):
                agent.learn(rotate(state, rotation), rotate(nextState, rotation), rotateAction(action, rotation), reward, isDone)

        prevState = state
        prevAction = action
        prevReward = reward
        isShouldLearn = True

    winner_history.append(env.checkWin())

# Check for the trained Q-table file
if not os.path.exists("qtable.pkl"):
    # Save the Q-table if it doesn't exist
    with open("qtable.pkl", "wb") as f:
        pickle.dump(qTable, f)

class TicTacToeGUI:
    def __init__(self, root, agent):
        self.root = root
        self.agent = agent
        self.env = Env()
        self.buttons = []
        self.create_board()
        self.reset_game()

        # Load the trained Q-table
        if os.path.exists("qtable.pkl"):
            with open("qtable.pkl", "rb") as f:
                self.agent.qTable = pickle.load(f)
        else:
            messagebox.showwarning("Warning", "Trained Q-table not found! Please train the model first.")

    def create_board(self):
        frame = tk.Frame(self.root)
        frame.pack()
        for i in range(9):
            button = tk.Button(frame, text="", font=('normal', 40), width=5, height=2,
                               command=lambda i=i: self.player_move(i))
            button.grid(row=i // 3, column=i % 3)
            self.buttons.append(button)

    def reset_game(self):
        self.env.reset()
        for button in self.buttons:
            button.config(text="", state=tk.NORMAL)

    def player_move(self, index):
        if self.env.board[index] == 0:
            self.env.board[index] = 1  # Player is 'X'
            self.update_board()
            if self.env.checkDone():
                self.end_game()
            else:
                self.agent_move()

    def agent_move(self):
        state = self.env.getState()
        action = self.agent.act(state)
        self.env.board[action] = 2  # Agent is 'O'
        self.update_board()
        if self.env.checkDone():
            self.end_game()

    def update_board(self):
        for i in range(9):
            symbol = "X" if self.env.board[i] == 1 else ("O" if self.env.board[i] == 2 else "")
            self.buttons[i].config(text=symbol)

    def end_game(self):
        winner = self.env.checkWin()
        if winner == 1:
            messagebox.showinfo("Result", "You win!")
        elif winner == 2:
            messagebox.showinfo("Result", "AI wins!")
        else:
            messagebox.showinfo("Result", "It's a draw!")
        self.reset_game()

# Start the GUI
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Tic Tac Toe with Q-learning")
    agent = Agent()
    gui = TicTacToeGUI(root, agent)
    root.mainloop()
