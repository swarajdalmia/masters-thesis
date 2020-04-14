import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patches as mpatches
from PIL import Image



circuit_dim = (64, 64)

# this uses emergent networks, that arise out of local reinforcement learning 
def main():
    # Some hyperparameters
    len_upper = []
    len_lower = []

    # 5*500 images will be stored 
    for j in range(5):
        prob_rand_global = 0.04*(j)
        prob_rand_local = 0.3 + 0.1*(j-1)
        for i in range(500):
            num = j*500 + i
            im_name = 'Image_'+str(num)+'.png'
            upper_bound_steps = 275  
            circuit = build_circuit_background()
            # display_circuit(circuit, 'Background Circuit')
            # reached stpres whether the circuit reached the output wire 

            circuit, reached_top = make_paths(circuit, [int(circuit_dim[0]/3 - 1),5], [int(circuit_dim[0]/3 - 1), circuit_dim[1]-6], 1, 8, upper_bound_steps, prob_rand_local, prob_rand_global)  # first input and first output 
            circuit, reached_bottom = make_paths(circuit, [int(2*circuit_dim[0]/3),5], [int(2*circuit_dim[0]/3), circuit_dim[1]-6] , 2, 9, upper_bound_steps, prob_rand_local, prob_rand_global) 
            # display_circuit(circuit, '')
            

            # if both circuits connected then save them and the lengths of both of them in the list 
            if(reached_top == True and reached_bottom == True):
                # save the circuits as is with legend and eveything else
                save_circuit(circuit, '/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/circuitImages/usefulCircuits/asIs_Legend/'+im_name)
                # remove the legends here
                save_circuit_simplified(circuit, '/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/circuitImages/usefulCircuits/asIs/'+im_name, True, True)
                # remove the noise but keep obstacles
                save_circuit_simplified(circuit, '/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/circuitImages/usefulCircuits/withObstacles_withoutNoise/'+im_name, True, False)
                # remove the obstacles but keep the noise 
                save_circuit_simplified(circuit, '/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/circuitImages/usefulCircuits/withoutObstacles_withNoise/'+im_name, False, True)
                a,b = path_lengths(circuit, 1,2)
                len_upper.append(a)
                len_lower.append(b)
                # greyscale with obstacles
                save_circuit_simplified(circuit, '/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/circuitImages/usefulCircuits/grey_withObstacles/'+im_name, True, False, True)
            else: 
                save_circuit(circuit, '/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/circuitImages/failedCircuits/'+im_name)

    # save the lengths of the successful circuits 
    len_upper = np.asarray(len_upper, dtype=np.int)
    len_lower = np.asarray(len_lower, dtype=np.int)
    np.savetxt('/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/circuitImages/usefulCircuits/len_upper.txt', len_upper, fmt='%d')
    np.savetxt('/Users/swarajdalmia/Desktop/3B/NeuroMorphicComputing/Code/circuitImages/usefulCircuits/len_lower.txt', len_lower, fmt='%d')

    # loading a np array 
    # b = np.loadtxt('test1.txt', dtype=int)

# this builds a basic background circuit. Adds inputs/outputs and obstacles and returns such an array 
# 0 - obstacles(green), 1 - top wire(white), 2 - bottom wire(blue), 5 - background(black), 6 - input(red), 7 - output(red)
# 8 - temp top wire(grey), 9 - temp bottom wire(grey)
def build_circuit_background():

    circuit = np.zeros(circuit_dim, dtype=int)
    circuit = np.array([x + 5 for x in circuit])  # these are the background

    # add inputs and outputs. The positions are handcrafted. The row placement is done to ensure equidistance and 
    # the columns are places to as to give as much possible space between the input/output while leaving some room 
    # behind 
    row = int(circuit_dim[0]/3 - 1)       # 128/3 - 1 = 41
    col = 4                               # 4
    circuit[row, col] =  circuit[row-1:row+2, col+1] = 6
    col = circuit_dim[1] - 5              # 123
    circuit[row, col] =  circuit[row-1:row+2, col-1] = 7 

    row = int(2*circuit_dim[0]/3)         # 85
    col = 4
    circuit[row, col] =  circuit[row-1:row+2, col+1] = 6
    col = circuit_dim[1] - 5
    circuit[row, col] =  circuit[row-1:row+2, col-1] = 7

    # add fixed obstacles 
    # circuit = place_obstacle(5,17, circuit)
    # circuit = place_obstacle(5,46, circuit)
    # circuit = place_obstacle(19,17, circuit)
    # circuit = place_obstacle(19,46, circuit)

    # add 8 variable obstacles of size 27*4
    ob_size = [20,3]
    for i in range(8):
        row = random.randrange(4 + int(ob_size[0]/2), circuit_dim[0]- 4 - int(ob_size[0]/2))   # randrange doesnt include the end but includes the start. This is the midrow of the obstacle
        col = 11 + int(i*5.6)                      
        circuit = place_obstacle(row, col, circuit, ob_size)
    return circuit

# returns the path lengths of the top and the bottom wire 
def path_lengths(circuit, top_wire_color, bottom_wire_color):
    unique, counts = np.unique(circuit, return_counts=True)
    counts = dict(zip(unique, counts))
    return counts[top_wire_color], counts[bottom_wire_color]

# row, col be mid position position for 2*9 obstacle
def place_obstacle(row, col, circuit, ob_size):
    circuit[row-int(ob_size[0]/2):row+int(ob_size[0]/2), col:col+ob_size[1]] =  0
    return circuit

def display_circuit(circuit, text, re_size = circuit_dim):
    # create discrete colormap
    cmap = colors.ListedColormap(['green', 'white', 'blue', 'black', 'red','red', 'darkgrey', 'darkgrey'])
    bounds = [0,1,2,5,6,7,8,9]
    labels = ['0','1','2','5','6','7','8','9']
    col = {1:'green', 2:'white', 3:'blue', 4:'black', 5:'red', 6:'darkgrey'}
    labels = {1:'Obstacles', 2:'Top Wire', 3:'Bottom Wire', 4:'Background', 5:'Input/Output', 6:'Unused Wire'}
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    ax.imshow(circuit, cmap=cmap, norm=norm, label = labels)
    ax.grid(which='major', axis='both', linestyle='-', color='grey', linewidth=1)
    patches =[mpatches.Patch(color=col[i],label=labels[i]) for i in col]    
    plt.legend(handles=patches, loc=1, borderaxespad=0., fontsize = 'xx-small')
    plt.title(text)
    plt.show()

def save_circuit(circuit, path, text = ""):
    # create discrete colormap
    cmap = colors.ListedColormap(['green', 'white', 'blue', 'black', 'red','red', 'darkgrey', 'darkgrey'])
    bounds = [0,1,2,5,6,7,8,9]
    labels = ['0','1','2','5','6','7','8','9']
    col = {1:'green', 2:'white', 3:'blue', 4:'black', 5:'red', 6:'darkgrey'}
    labels = {1:'Obstacles', 2:'Top Wire', 3:'Bottom Wire', 4:'Background', 5:'Input/Output', 6:'Unused Wire'}
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots()
    ax.imshow(circuit, cmap=cmap, norm=norm, label = labels)
    ax.grid(which='major', axis='both', linestyle='-', color='grey', linewidth=1)
    patches =[mpatches.Patch(color=col[i],label=labels[i]) for i in col]    
    plt.legend(handles=patches, loc=1, borderaxespad=0., fontsize = 'xx-small')
    plt.title(text)
    plt.savefig(path, bbox_inches='tight')


def save_circuit_simplified(circuit, path, display_obstacles = True, display_noise = True, gre_scale = False):
    # create discrete colormap
    if(display_obstacles == False and display_noise == False):
        cmap = colors.ListedColormap(['black', 'white', 'blue', 'black', 'red','red', 'black', 'black'])
    elif(display_obstacles == True and display_noise == False):
        cmap = colors.ListedColormap(['green', 'white', 'blue', 'black', 'red','red', 'black', 'black'])
    elif(display_obstacles == False and display_noise == True):
        cmap = colors.ListedColormap(['black', 'white', 'blue', 'black', 'red','red', 'darkgrey', 'darkgrey'])
    else:
        cmap = colors.ListedColormap(['green', 'white', 'blue', 'black', 'red','red', 'darkgrey', 'darkgrey'])
    
    if(gre_scale == True):
        cmap = colors.ListedColormap(['grey', 'white', 'white', 'black', 'white','white', 'black', 'black'])

    bounds = [0,1,2,5,6,7,8,9]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.axis('off')
    plt.legend('off')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(circuit, cmap=cmap, norm=norm)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.savefig(path, bbox_inches='tight', pad_inches = 0)

# makes paths to connect the respective inputs to the respective outputs using probabalistic reinforcement
# learning, the start pos and the end pos are the mid circuit elements. 
def make_paths(circuit, start_pos, end_pos, color_wire, color_temp_wire, upper_bound_steps, prob_rand_local, prob_rand_global):
    # loop till a certain upper_bound_steps or till the output has been reached 
    i = 0 
    curr_pos = start_pos
    # for the 1st iteration simply move ahead and add a wire
    curr_pos[1] = curr_pos[1]+1 
    circuit[curr_pos[0],curr_pos[1]] = color_wire    
    reached = False
    # at this stage there is a wire in curr_pos 
    while (i <= upper_bound_steps):
        # add the next wire starting from curr_pos and update curr_pos
        circuit, curr_pos = add_next_wire(circuit, curr_pos, end_pos, color_wire, color_temp_wire, prob_rand_local, prob_rand_global)
        # check if output wire has been reached, if it is somewhat close, otherwise skip to ensure faster running
        if(i >= 35 and calc_distance(curr_pos, end_pos)<3 and reached_output(curr_pos, end_pos)):
                reached = True
                break
        # if(i%20 == 0):
        #     display_circuit(circuit, "Circuit after "+str(i)+" iterations")
        i+=1
    
    # display_circuit(circuit, "Final Circuit after "+str(i)+" iterations")
    return circuit, reached 

# calculates shortest manhattan distance 
def calc_distance(curr_pos, end_pos):
    return abs(curr_pos[0]-end_pos[0]) + abs(curr_pos[1]-end_pos[1])

# calculates euclidean distance 
def calc_euclidean_dist(curr_pos, end_pos):
    curr_pos = np.array(curr_pos)
    end_pos =np.array(end_pos)
    return np.linalg.norm(curr_pos - end_pos)

# checks if the curr_pos wire is connected to the output wire(specified by end_pos)
def reached_output(curr_pos, end_pos):
    row = end_pos[0]
    col = end_pos[1]
    if(calc_distance(curr_pos, end_pos) == 1 or calc_distance(curr_pos, [row, col+1]) == 1 or 
    calc_distance(curr_pos, [row-1, col]) == 1 or calc_distance(curr_pos, [row+1, col])==1):
        return True
    return False

# randomly adds a global circuit element of type wire_color, the gloabl is in the top or bottom half depending on where we go !
def add_global_random_wire(circuit, color_temp_wire):
    row_lims = [0,circuit_dim[0]]

    if(color_temp_wire == 8):
        row_lims[1] = circuit_dim[0]/2
    else:
        row_lims[0] = circuit_dim[0]/2

    row_random = random.randrange(row_lims[0], row_lims[1]) 
    col_random = random.randrange(0, circuit_dim[1]) 
    i = 0
    while(circuit[row_random, col_random] != 5 and i <=5):
        row_random = random.randrange(row_lims[0], row_lims[1]) 
        col_random = random.randrange(0, circuit_dim[1]) 
        i+= 1     
    if(circuit[row_random, col_random] == 5):
        circuit[row_random, col_random]= color_temp_wire
    return circuit

# randomly adds a local circuit element either in the forward or backbard direction, which ever takes it closer to end_pos
def add_local_random_wire(circuit, color_temp_wire, curr_pos, end_pos):
    found = False
    i = 0
    n_row = n_col = 0
    sign = 1
    if (curr_pos[1] >= end_pos[1] + 2):
        sign = -1
    while(found != True and i <=10):
        i = i+1
        n_row = curr_pos[0] + random.randint(0, 4) - 2
        n_col = curr_pos[1] + random.randint(0, 2)*sign
        # if blackground found and if its a valid entry
        if (n_row>=0 and n_row <circuit_dim[0] and n_col>=0 and n_col<circuit_dim[1] and circuit[n_row, n_col] == 5): 
            found = True
    if(found == True):
        circuit[n_row,n_col] = color_temp_wire # random circuit element added
        # print("random element added at", n_row, n_col)
    return circuit

# returns true if addition of temp wire to pos_to_eval, connects to further temp wires. Else returns false
# at this point it is known that pos_to_eval contains empty space
def adds_further_extention(circuit, pos_to_eval, color_temp_wire):
    row = pos_to_eval[0]
    col = pos_to_eval[1]
    if((row+1 <circuit_dim[0] and circuit[row+1, col] == color_temp_wire) or (row - 1 >=0 and circuit[row-1, col] == color_temp_wire) 
        or (col+1 <circuit_dim[1] and circuit[row, col+1] == color_temp_wire) or (col-1 >=0 and circuit[row, col-1] == color_temp_wire)):
        return True
    return False

# add a conecting wire in one of the 4 directions (considering distance and further connectivity) and considering there is space
def reward_based_extention(circuit, color_wire, color_temp_wire, curr_pos, end_pos):

    weight = random.uniform(0.5, 1.25) # the distance metric adds distance 1 based on manhattan distance 
    curr_row = curr_pos[0]
    curr_col = curr_pos[1]
    # array [_,_,_,_] stores the weight of going forward, bakcward, top, down 
    # print("moving either forward/backward/top/down")
    # print("curr_pos is ", curr_pos)
    direction = [0,0,0,0]
    # Weight for FORWARD direction ! 
    # obstacles and circuit bounds considered and weights added for distance
    if(curr_col+1>=circuit_dim[1] or circuit[curr_row, curr_col+1] != 5):  #if going forward is not free or one cant go forward
        direction[0] = -99
    else: 
        direction[0] = calc_euclidean_dist(curr_pos, end_pos) - calc_euclidean_dist([curr_row, curr_col+1], end_pos)
        # add weight if it connects to something forward. Add prob weight to ensure variability to metric 
        if(adds_further_extention(circuit, [curr_row, curr_col+1], color_temp_wire)):
            direction[0] += weight

    if(curr_col-1<0 or circuit[curr_row, curr_col-1] != 5):  # going back
        direction[1] = -99
    else: 
        direction[1] = calc_euclidean_dist(curr_pos, end_pos) - calc_euclidean_dist([curr_row, curr_col-1], end_pos)
        # add weight if it connects to something forward. Add prob weight to ensure variability to metric 
        if(adds_further_extention(circuit, [curr_row, curr_col-1], color_temp_wire)):
            direction[1] += weight

    if(curr_row == 0 or circuit[curr_row-1, curr_col] != 5):  # going up 
        direction[2] = -99
    else:
        direction[2] = calc_euclidean_dist(curr_pos, end_pos) - calc_euclidean_dist([curr_row-1, curr_col], end_pos)
        if(adds_further_extention(circuit, [curr_row-1, curr_col], color_temp_wire)):
            direction[2] += weight

    if(curr_row+1>=circuit_dim[0] or circuit[curr_row+1, curr_col] != 5): # going down
        direction[3] = -99
    else:
        direction[3] = calc_euclidean_dist(curr_pos, end_pos) - calc_euclidean_dist([curr_row+1, curr_col], end_pos)
        if(adds_further_extention(circuit, [curr_row+1, curr_col], color_temp_wire)):
            direction[3] += weight

    # print("direction metric", direction)
    # find action with largest weight in random order and perform the action
    fs = [forward_if_best, backward_if_best, up_if_best, down_if_best]
    random.shuffle(fs)
    found = False
    for f in fs:
        arg = [direction, circuit, color_temp_wire, curr_row, curr_col]
        circuit, found = f(arg[0],arg[1],arg[2],arg[3],arg[4])
        if(found):
            break
    return circuit

# if forward has highest weight in direction vector then go forward
def forward_if_best(direction, circuit, color_temp_wire, curr_row, curr_col):
    # print("Tried forward")
    if(direction[0] >= direction[1] and direction[0] >= direction[2] and direction[0] >= direction[3] and direction[0]>-0.5):
        circuit[curr_row, curr_col + 1] = color_temp_wire
        return circuit, True
    return circuit, False

def backward_if_best(direction, circuit, color_temp_wire, curr_row, curr_col):
    # print("Tried backward")
    if(direction[1] >= direction[0] and direction[1] >= direction[2] and direction[1] >= direction[3] and direction[1]>-0.5):
        circuit[curr_row, curr_col - 1] = color_temp_wire
        return circuit, True
    return circuit, False

def up_if_best(direction, circuit, color_temp_wire, curr_row, curr_col):
    # print("Tried up")
    if(direction[2] >= direction[0] and direction[2] >= direction[1] and direction[2] >= direction[3] and direction[2]>-0.5):
        circuit[curr_row-1, curr_col] = color_temp_wire
        return circuit, True
    return circuit, False

def down_if_best(direction, circuit, color_temp_wire, curr_row, curr_col):
    # print("Tried down")
    if(direction[3] >= direction[0] and direction[3] >= direction[1] and direction[3] >= direction[2] and direction[3]>-0.5):
        circuit[curr_row+1, curr_col] = color_temp_wire
        return circuit, True
    return circuit, False

# add next wire using reinforcement learning. Is it actually reinforcement learning ?
def add_next_wire(circuit, curr_pos, end_pos, color_wire, color_temp_wire, prob_rand_local, prob_rand_global):
    prob = random.uniform(0, 1)
    # This block adds a temp wire which is always present
    # randomly add a global circuit element of type wire_color
    if(prob < prob_rand_global):
        # print('adds a random global wire')
        circuit = add_global_random_wire(circuit, color_temp_wire)
    # randomly add a wire in the neighbourhood of curr_pos. Add wire in the forward or backward direction depending on where end_pos
    elif(prob < prob_rand_local+prob_rand_global): 
        # print('adds a random local wire')
        circuit = add_local_random_wire(circuit, color_temp_wire, curr_pos, end_pos)
    # add a conecting wire in one of the 4 directions (considering distance and further connectivity) and considering there is space
    else: 
        # print('adds an extention wire to extend the path')
        circuit = reward_based_extention(circuit, color_wire, color_temp_wire, curr_pos, end_pos)
    # find leaf finalised the temp wires to the color_wire and and reached the leaf node
    # display_circuit(circuit, "before finding leaf")
    circuit, n_row, n_col = find_next_leaf(circuit, curr_pos[0], curr_pos[1], color_wire, color_temp_wire, end_pos)
    # update position of curr_pos after adding wire
    return circuit, [n_row, n_col]

# finds the next start position 
def find_next_leaf(circuit, curr_row, curr_col, color_wire, color_temp_wire, end_pos):
    # print("finding next leaf")
    # print("current leaf is at", curr_row, curr_col)
    fs = [go_foward, go_backward, go_down, go_up]
    evaluation = True
    while(evaluation):
        # randomise which direction it checks initially
        evaluation = False
        random.shuffle(fs)
        for f in fs:
            arg = [circuit, curr_row, curr_col, color_wire, color_temp_wire]
            circuit, curr_row, curr_col, moved = f(arg[0],arg[1],arg[2],arg[3],arg[4])
            evaluation = (evaluation or moved)   # even if there is one move, this stores true
        if(reached_output([curr_row,curr_col], end_pos)):
            break
    # print("found next leaf")
    # print("final leaf is at", curr_row, curr_col)
    return circuit, curr_row, curr_col

def go_foward(circuit, curr_row, curr_col, color_wire, color_temp_wire):
    moved = False
    if(curr_col + 1<circuit_dim[1] and circuit[curr_row, curr_col + 1] == color_temp_wire):
        circuit[curr_row, curr_col + 1] = color_wire
        curr_col += 1
        moved = True
        # print("leaf moved forward")
    return circuit, curr_row, curr_col, moved

def go_backward(circuit, curr_row, curr_col, color_wire, color_temp_wire):
    moved = False
    # the less than 43 is added so as to prevent circularity is as many situations as we can avoid. 
    if(curr_col - 1 >=0  and curr_col >= circuit_dim[1]-1 and circuit[curr_row, curr_col - 1] == color_temp_wire):
        circuit[curr_row, curr_col - 1] = color_wire
        curr_col += -1
        moved = True
        # print("leaf moved backward")
    return circuit, curr_row, curr_col, moved

def go_up(circuit, curr_row, curr_col, color_wire, color_temp_wire):
    moved = False
    if(curr_row-1>=0 and circuit[curr_row-1, curr_col] == color_temp_wire):
        circuit[curr_row-1, curr_col] = color_wire
        curr_row += -1
        moved = True
        # print("leaf moved up")
    return circuit, curr_row, curr_col, moved

def go_down(circuit, curr_row, curr_col, color_wire, color_temp_wire):
    moved = False
    if(curr_row+1<circuit_dim[0] and circuit[curr_row+1, curr_col] == color_temp_wire):
        circuit[curr_row+1, curr_col] = color_wire
        curr_row += 1
        moved = True
        # print("leaf moved down")
    return circuit, curr_row, curr_col, moved

if __name__ == '__main__':
    main()
    print(1)