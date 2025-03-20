# Napoleon Tic-Tac-Toe

Napoleon Tic-Tac-Toe is an AI-powered agent designed to compete in **K-in-a-Row**, a generalized Tic-Tac-Toe game with customizable board sizes, win conditions, and forbidden squares. Developed for an AI course project, Napoleon uses **minimax search with alpha-beta pruning** to evaluate moves efficiently and dynamically adapts to opponents. The agent also engages in **interactive dialogue**, maintaining a persona while providing in-game commentary.

---

## Features  
- **Advanced AI Strategy** – Implements the minimax algorithm for decision-making.  
- **Alpha-Beta Pruning** – Improves search efficiency by eliminating unnecessary computations.  
- **Interactive Persona** – Generates dynamic responses during gameplay to enhance engagement.  
- **Adaptable Playstyle** – Competes on different board configurations and game types.  

---

AI IMPLEMENTATION

Napoleon uses minimax search with alpha-beta pruning to explore potential moves efficiently. 
The AI evaluates board states using a custom static evaluation function, prioritizing winning sequences while blocking opponents. 
Additional heuristics such as board control and move weighting enhance decision-making.

---

Tournament Performance
Napoleon competed in the K-in-a-Row agent tournament and tied for second place, demonstrating strong strategic play against other AI agents.

---

## Installation & Setup  

### Prerequisites  
- Python 3.x  
- Required libraries: `numpy`, `pygame` (if visualization is used)  

### Running the Agent  
1. Clone the repository:  
   ```sh
   git clone your-repo-url napoleon-tic-tac-toe
   cd napoleon-tic-tac-toe
---

Install Dependencies
pip install -r requirements.txt

---

Run The Game Engine:
python GameMasterOffline.py
