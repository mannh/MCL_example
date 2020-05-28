# Monte Carlo Localization of a Mobile Robot Using Landmarks Example

This assignment is taken from course Introduction to Robtics (Ben Gurion University).

Consider a planar robot with three DOFs x = (x, y, θ) operating in a world of size 100 × 100.   
The world includes six landmarks at  
m = {(20, 20),(50, 20),(80, 20),(80, 80),(50, 80),(20, 80)}

The robot can take two motor commands, a turn movement command u1,(u1 ∈ [0, 2π)) and a forward  
movement command u2 (u2 > 0). The deterministic motion model of the robot is given by  
θ' = 0 = θ + u1   
x' = x + u2 cos θ'  
y' = y + u2 sin θ'  

