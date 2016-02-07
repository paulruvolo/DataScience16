import cv2
import numpy as np
from copy import deepcopy

canvas_dim = 500

def make_actual_tree():
    tree = DecisionTree()
    make_actual_tree_helper(tree.decision_node, 1)
    return tree

def make_actual_tree_helper(node, split_prob=.5):
    is_vertical = np.random.randint(0, 2)
    gt_label = np.random.randint(0, 2) == 1

    if is_vertical:
        r = node.x_bounds[1] - node.x_bounds[0]
    else:
        r = node.y_bounds[1] - node.y_bounds[0]
    if r > 250:
        split_prob = 1
    elif r > 100:
        split_prob = .4
    elif r < 5:
        split_prob = 0

    if np.random.rand() > split_prob:
        return
    if is_vertical:
        x = np.random.randint(node.x_bounds[0], node.x_bounds[1])
        y = None
    else:
        x = None
        y = np.random.randint(node.y_bounds[0], node.y_bounds[1])
    node.set_formula(x, y, is_vertical, gt_label)
    make_actual_tree_helper(node.true_node)
    make_actual_tree_helper(node.false_node)

class DecisionTree(object):
    def __init__(self):
        self.decision_node = DecisionNode()
        self.decision_node.label = False

    def evaluate(self, x, y):
        return self.decision_node.evaluate(x, y)

    def __str__(self):
        return str(self.decision_node)

    def draw(self, im, thickness=1):
        self.decision_node.draw(im, thickness)

class DecisionNode(object):
    def __init__(self):
        self.false_node = None
        self.true_node = None
        self.label = None
        self.formula = None
        self.nesting = 0
        self.x_bounds = [0, canvas_dim]
        self.y_bounds = [0, canvas_dim]

    def evaluate(self, x, y):
        if self.label != None:
            return self.label, self
        decision = eval(self.formula)
        if decision:
            return self.true_node.evaluate(x, y)
        else:
            return self.false_node.evaluate(x, y)

    def set_formula(self, x, y, is_vertical, gt_label):
        self.label = None

        if is_vertical:
            self.formula = "x > %f" % (x,)
        else:
            self.formula = "y > %f" % (y,)
        self.true_node = DecisionNode()
        self.false_node = DecisionNode()

        self.true_node.label = gt_label
        self.true_node.nesting = self.nesting + 1
        self.false_node.label = not gt_label
        self.false_node.nesting = self.nesting + 1

        self.true_node.x_bounds = self.x_bounds[:]
        self.true_node.y_bounds = self.y_bounds[:]

        self.false_node.x_bounds = self.x_bounds[:]
        self.false_node.y_bounds = self.y_bounds[:]

        if is_vertical:
            self.true_node.x_bounds[0] = x
            self.false_node.x_bounds[1] = x
        else:
            self.true_node.y_bounds[0] = y
            self.false_node.y_bounds[1] = y

    def draw(self, im, thickness=1):
        if self.label != None:
            color_map = {True:(1,0,0), False:(0,0,1)}
            cv2.rectangle(im,
                          (self.x_bounds[0], self.y_bounds[0]),
                          (self.x_bounds[1], self.y_bounds[1]),
                          color_map[self.label],
                          thickness=-1)                
            if thickness > 0:
                cv2.rectangle(im,
                              (self.x_bounds[0], self.y_bounds[0]),
                              (self.x_bounds[1], self.y_bounds[1]),
                              (0,0,0),
                              thickness=thickness)  
        else:
            self.true_node.draw(im, thickness)
            self.false_node.draw(im, thickness)

    def __str__(self):
        return_value = ""
        if self.label != None:
            return_value += "    "*self.nesting + str(self.label) + "\n"
        else:
            return_value += "    "*self.nesting + "if %s\n" % (self.formula)
            return_value += str(self.true_node)
            return_value += "    "*self.nesting + "else\n"
            return_value += str(self.false_node)
        return return_value


def draw_line(event,x,y,flags,param):
    global I_display
    global decision_tree
    global gt_label
    global x_last
    global y_last
    x_last = x
    y_last = y


    I_display = np.copy(I)
    if is_vertical:
        cv2.line(I_display, (x,0), (x, I_display.shape[0]), (0))
    else:
        cv2.line(I_display, (0,y), (I_display.shape[1], y), (0))

    if event == cv2.EVENT_RBUTTONDOWN:
        gt_label = not gt_label

    if event == cv2.EVENT_LBUTTONDBLCLK:
        _, decision_node = decision_tree.evaluate(x, y)
        decision_node.set_formula(x, y, is_vertical, gt_label)
        decision_tree.draw(I_display, thickness=1)
        print decision_tree
    else:
        decision_tree_copy = deepcopy(decision_tree)
        _, decision_node = decision_tree_copy.evaluate(x, y)
        decision_node.set_formula(x, y, is_vertical, gt_label)
        decision_tree_copy.draw(I_display, thickness=1)
    I_display = cv2.addWeighted(I_display, 0.5, I, 0.5, 0)

x_last = 0
y_last = 0
is_vertical = True
gt_label = True

decision_tree = DecisionTree()
cv2.namedWindow('points')
cv2.setMouseCallback('points', draw_line)
real_tree = make_actual_tree()

print real_tree

n_points = 50
noise = .2
points = np.random.randint(0, canvas_dim, (n_points, 2))
labels = []
for i in range(n_points):
    label, _ = real_tree.evaluate(points[i,0], points[i,1])
    if np.random.rand() < noise:
        label = not label
    labels.append(label)

I = np.ones((canvas_dim, canvas_dim, 3))
for i in range(n_points):
    if labels[i]:
        cv2.putText(I, "+", (points[i,0], points[i,1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 2)
    else:
        cv2.putText(I, "-", (points[i,0], points[i,1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 2)

I_display = np.copy(I)

while True:
    key = cv2.waitKey(30)
    if key == ord('q'):
        break
    if key == ord(' '):
        is_vertical = not is_vertical
        draw_line(None, x_last, y_last, None, None)
    cv2.imshow('points', I_display)
    I_display_true = np.copy(I)
    real_tree.draw(I_display_true, thickness=0)
    cv2.imshow('true', I_display_true)
    I_guess = np.ones((canvas_dim, canvas_dim, 3))
    decision_tree.draw(I_guess, thickness=0)
    blend = I_display_true * I_guess
    blend_two_channel = blend.max(axis=2)
    cv2.imshow('comparison', blend_two_channel)