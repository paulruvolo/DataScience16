"""
    An interactive application to demonstrate the complexities that
    arise in creating decision trees.

    Once the application has been started, you will see a window containing
    some labeled training data (+ for positive, - for negative).  You can
    then create a decision tree to classify this data by double clicking
    in the window.  By double clicking you will create a split on either the
    x or y value.  You can tell which one the split will use by the orientation
    of the line that will preview the split you are about to create.  Use the
    spacebar to toggle the split direction.

    Which label to classify the points on either side of the new split can be
    changed by right clicking in the window.

    Once you are satisfied with your tree, tap 'q' to see how your decision
    tree compares to the tree decision function.

    To explore different scenarios, you can edit the variables
    canvas_dim: the dimensionality of the canvas (must be square)
    n_points: the number of training points
    noise: the probability of the true label being flipped in the training set.
"""

import cv2
import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from numpy import meshgrid
import matplotlib.pyplot as plt

canvas_dim = 500
n_points = 100
noise = .2

def make_random_tree():
    """ Return a random decision tree """
    tree = DecisionTree()
    make_random_tree_helper(tree.decision_node)
    return tree

def make_random_tree_helper(node):
    """ A helper function to create random tree rooted at
        the `node`.  The split probability is determined
        based on how large of a section of the canvas
        is encompassed by the node. """
    is_vertical = np.random.randint(0, 2)
    gt_label = np.random.randint(0, 2) == 1

    if is_vertical:
        r = node.x_bounds[1] - node.x_bounds[0]
    else:
        r = node.y_bounds[1] - node.y_bounds[0]
    if r > canvas_dim*0.5:
        split_prob = 1
    elif r > canvas_dim*0.2:
        split_prob = .4
    elif r < canvas_dim*0.01:
        split_prob = 0
    else:
        split_prob = 0.5

    if np.random.rand() > split_prob:
        return

    if is_vertical:
        x = np.random.randint(node.x_bounds[0], node.x_bounds[1])
        y = None
    else:
        x = None
        y = np.random.randint(node.y_bounds[0], node.y_bounds[1])
    node.set_formula(x, y, is_vertical, gt_label)

    # recurse and create the subtrees when the test evaluates to
    # True or False
    make_random_tree_helper(node.true_node)
    make_random_tree_helper(node.false_node)

class DecisionTree(object):
    """ A Decision tree class.  This class is a thin interface
        on top of the DecisionNode which does most of the work """
    def __init__(self):
        self.decision_node = DecisionNode()
        self.decision_node.label = False

    def evaluate(self, x, y):
        """ return the truth value for the specified x, y pair """
        return self.decision_node.evaluate(x, y)

    def __str__(self):
        return str(self.decision_node)

    def draw(self, im, thickness=1):
        """ Render the decision tree in the specified image where
            red indicates regions of the image that will evaluate
            as False, and blue for the regions of the image that
            will evaluate to True. """
        self.decision_node.draw(im, thickness)

class DecisionNode(object):
    """ A node in the decision tree

        false_node: the decision_node to use if the formula evaluates
                    to False.  This will be None if this is a terminal
                    node.
        true_node: the decision_node to use if the formula evaluates
                   to False.  This will be None if this is a terminal
                   node.
        label: the Boolean value to return.  If this node is not a terminal
               node then label will be None.
        formula: this is a Boolean formula involving the variables x and y
                 that determines which sub tree to use for further evaluation.
                 For terminal nodes, thsi will be None.
        nesting: the nesting of the node from the root (if root nesting=0).
                 This is only used for formatting in the __str__ function.
        x_bounds: the left and right edge of the rectangle that falls within
                  this node.  This is only used for rendering the visual version
                  of the tree.
        y_bounds: the top and bottom edge of the rectangle that falls within
                  this node.  This is only used for rendering the visual version
                  of the tree.
    """
    def __init__(self):
        self.false_node = None
        self.true_node = None
        self.label = None
        self.formula = None
        self.nesting = 0
        self.x_bounds = [0, canvas_dim]
        self.y_bounds = [0, canvas_dim]

    def evaluate(self, x, y):
        """ Return the Boolean value for this x, y pair when starting
            evaluation from this node.  """
        if self.label != None:
            return self.label, self
        decision = eval(self.formula)
        if decision:
            return self.true_node.evaluate(x, y)
        else:
            return self.false_node.evaluate(x, y)

    def set_formula(self, x, y, is_vertical, gt_label):
        """ Convert a terminal node to an intermediate node.

            x: is the split point when is_vertical is True
            y: is the split point when is_vertical is False
            is_vertical: True if splitting on x, False if splitting on y
            gt_label: The label when the appropriate variable is greater than
                      the split point.  This is used to create the true and
                      false subtrees.
        """

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
        """ Render the decision node graphically on the specified image, im.
            The argument thickness controls the border drawn around the
            rectangular decision regions. """
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
            return_value += "    "*self.nesting + "return " + str(self.label) + "\n"
        else:
            return_value += "    "*self.nesting + "if %s:\n" % (self.formula)
            return_value += str(self.true_node)
            return_value += "    "*self.nesting + "else:\n"
            return_value += str(self.false_node)
        return return_value


def draw_line(event,x,y,flags,param):
    """ Process mouse events so we can get a preview of the split
        that would be added if user double clicks.  This also will
        create a split when the user double clicks. """
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
cv2.namedWindow('guess')
cv2.setMouseCallback('guess', draw_line)
real_tree = make_random_tree()

print real_tree

points_all = np.random.randint(0, canvas_dim, (n_points*100, 2))
labels_all = np.zeros((points_all.shape[0],),dtype=np.bool_)
for i in range(len(labels_all)):
    label, _ = real_tree.evaluate(points_all[i,0], points_all[i,1])
    if np.random.rand() < noise:
        label = not label
    labels_all[i] = label

points, points_test, labels, labels_test = train_test_split(points_all, labels_all, train_size=0.1)
I = np.ones((canvas_dim, canvas_dim, 3))
for i in range(n_points):
    if labels[i]:
        (x_size, y_size), _ = cv2.getTextSize("+", cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
        cv2.putText(I, "+", (points[i,0]-x_size/2, points[i,1]+y_size/2), cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 2)
    else:
        (x_size, y_size), _ = cv2.getTextSize("-", cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
        cv2.putText(I, "-", (points[i,0]-x_size/2, points[i,1]+y_size/2), cv2.FONT_HERSHEY_COMPLEX, 0.5, 0, 2)

I_display = np.copy(I)

while True:
    key = cv2.waitKey(30)
    if key == ord('q'):
        break
    if key == ord(' '):
        is_vertical = not is_vertical
        draw_line(None, x_last, y_last, None, None)
    cv2.imshow('guess', I_display)

# render the true decision tree
I_display_true = np.ones((canvas_dim, canvas_dim, 3))
real_tree.draw(I_display_true, thickness=0)
#cv2.imshow('true', I_display_true)

# render the user's constructed decision tree
I_guess = np.ones((canvas_dim, canvas_dim, 3))
decision_tree.draw(I_guess, thickness=0)
#cv2.imshow('guess', I_guess)

# create a blend of the two that is black when they disagree and
# white when the two trees agree.
# blend = I_display_true * I_guess
# blend_two_channel = blend.max(axis=2)
# cv2.imshow('comparison', blend_two_channel)

predictions = np.asarray([real_tree.evaluate(pt[0], pt[1])[0] for pt in points_test])
print "True model's accuracy", np.mean(predictions == labels_test)

predictions = np.asarray([decision_tree.evaluate(pt[0], pt[1])[0] for pt in points_test])
print "Your model's accuracy", np.mean(predictions == labels_test)

model = DecisionTreeClassifier()
model.fit(points, labels)
print "Sklearn Single Decision Tree accuracy", model.score(points_test, labels_test)

x, y = meshgrid(range(canvas_dim),range(canvas_dim))
get_decision_func = np.hstack((x.reshape(x.shape[0]*x.shape[1],1), y.reshape(y.shape[0]*y.shape[1],1)))

decision_func_output = model.predict(get_decision_func).astype(dtype=np.float).reshape((canvas_dim, canvas_dim))
tree_func_color = np.zeros((canvas_dim, canvas_dim, 3))
tree_func_color[:,:,0] = decision_func_output
tree_func_color[:,:,2] = 1 - decision_func_output
#cv2.imshow("Decision Tree Decision", tree_func_color)

model = RandomForestClassifier()
model.fit(points, labels)
print "Sklearn Random Forest accuracy", model.score(points_test, labels_test)

decision_func_output = model.predict(get_decision_func).astype(dtype=np.float).reshape((canvas_dim, canvas_dim))
forest_func_color = np.zeros((canvas_dim, canvas_dim, 3))
forest_func_color[:,:,0] = decision_func_output
forest_func_color[:,:,2] = 1 - decision_func_output
#cv2.imshow("Forest Decision", forest_func_color)
cv2.destroyAllWindows()

plt.subplot(2,2,1)
plt.title("True Decision Function")
plt.imshow(np.flipud(cv2.cvtColor(I_display_true.astype(np.uint8)*255, cv2.COLOR_BGR2RGB)), origin="lower")
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Your Decision Function")
plt.imshow(np.flipud(cv2.cvtColor(I_guess.astype(np.uint8)*255, cv2.COLOR_BGR2RGB)), origin="lower")
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Sklearn Single Tree Decision Function")
plt.imshow(np.flipud(cv2.cvtColor(tree_func_color.astype(np.uint8)*255, cv2.COLOR_BGR2RGB)), origin="lower")
plt.axis('off')

plt.subplot(2,2,4)
plt.title("Sklearn Forest Decision Function")
plt.imshow(np.flipud(cv2.cvtColor(forest_func_color.astype(np.uint8)*255, cv2.COLOR_BGR2RGB)), origin="lower")
plt.axis('off')

plt.show()