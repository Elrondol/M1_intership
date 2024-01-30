import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def compute_final_positions():
    final_positions = np.zeros((8,5,5))
    
    final_positions[0,:,:] = np.array([[7, 7,0,2,2 ],
                                        [0,0,3,2,1],
                                        [0,3,3,4,1],
                                        [5,9,9,4,8],
                                        [0,6,8,8,8]])
    
    final_positions[1,:,:] = np.array([[9, 9,0,2,2 ],
                                        [0,0,3,2,1],
                                        [0,3,3,4,1],
                                        [5,7,7,4,8],
                                        [0,6,8,8,8]])
    
    final_positions[2,:,:] = np.array([[7, 7,0,2,2 ],
                                        [0,0,3,2,4],
                                        [0,3,3,1,4],
                                        [5,9,9,1,8],
                                        [0,6,8,8,8]])
    
    final_positions[3,:,:] = np.array([[9, 9,0,2,2 ],
                                        [0,0,3,2,4],
                                        [0,3,3,1,4],
                                        [5,7,7,1,8],
                                        [0,6,8,8,8]])
    
    final_positions[4,:,:] = np.array([[7, 7,0,2,2 ],
                                        [0,0,3,2,1],
                                        [0,3,3,4,1],
                                        [6,9,9,4,8],
                                        [0,5,8,8,8]])
    
    final_positions[5,:,:] = np.array([[9, 9,0,2,2 ],
                                        [0,0,3,2,1],
                                        [0,3,3,4,1],
                                        [6,7,7,4,8],
                                        [0,5,8,8,8]])
    
    final_positions[6,:,:] = np.array([[9, 9,0,2,2 ],
                                        [0,0,3,2,4],
                                        [0,3,3,1,4],
                                        [6,7,7,1,8],
                                        [0,5,8,8,8]])
    
    final_positions[7,:,:] = np.array([[7, 7,0,2,2 ],
                                        [0,0,3,2,4],
                                        [0,3,3,1,4],
                                        [6,9,9,1,8],
                                        [0,5,8,8,8]])
    
    return final_positions


def plot_board(positions):
        # Define colors for different values
    color_map = {0: 'white', 1: 'orange', 4: 'orange', 2: 'green',
                 3: 'red', 5: 'blue', 6: 'blue', 7: 'yellow',
                 9: 'yellow', 8: 'purple'}

    # Create a colormap with custom colors
    cmap = colors.ListedColormap([color_map[i] for i in range(10)])

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the array using imshow
    im = ax.imshow(positions, cmap=cmap)

    # Set ticks and labels
    ax.set_xticks(np.arange(positions.shape[1]))
    ax.set_yticks(np.arange(positions.shape[0]))
    ax.set_xticklabels(np.arange(positions.shape[1]))
    ax.set_yticklabels(np.arange(positions.shape[0]))

    # Set the tick positions to be in the center of the cells
    ax.set_xticks(np.arange(positions.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(positions.shape[0] + 1) - 0.5, minor=True)

    # Set the aspect ratio to be equal and adjust the limits
    ax.set_aspect('equal')
    ax.set_xlim([-0.5, positions.shape[1] - 0.5])
    ax.set_ylim([-0.5, positions.shape[0] - 0.5])
    for i in range(5):
        for j in range(5):
            if positions[4-i,4-j]!=0:
                plt.text(4-j,4-i, f'{int(positions[4-i,4-j])}')
    plt.gca().invert_yaxis()


    
def compute_new_positions(current_board):
    new_positions = [] 
    score_list = []
    move_list = []
    for i in range(1,10):
        piece_idx = np.where(current_board==i) #on trouve la loc des partie de la piece
        score = len(piece_idx[0][:]) #compte le nombre de blocs = score
        new_piece_idx = []
        for z in range(len(piece_idx[0])): #pour aller up c'est -1 et  pour aller right c'est +1 et left c'ets -1 !!!   on itère sur les blocs d'une piece
            new_piece_idx.append([piece_idx[0][z]-1,piece_idx[1][z]]) #UP
            new_piece_idx.append([piece_idx[0][z]+1,piece_idx[1][z]]) #DOWN
            new_piece_idx.append([piece_idx[0][z],piece_idx[1][z]-1]) #LEFT 
            new_piece_idx.append([piece_idx[0][z],piece_idx[1][z]+1]) #RIGHT
        #maintenant qu'on connait les nouveaux indices de la piece dans les 4 directions on peut vérifier si elle peut boucler dans les diverses directions
        new_piece_idx = np.array(new_piece_idx)

        move_up = True
        move_down = True
        move_left = True
        move_right = True

        for j in np.arange(0,len(new_piece_idx[:,0]),4): #tous les 4 car tous les 4 = un nouveau bloc considéré
            #pour le up on vérifie si à la nouvelle position du bloc après déplacement il y avait de base un 0 ou même val
            if (0<=new_piece_idx[j,0]<=4) and  (0<=new_piece_idx[j,1]<=4):
                if current_board[new_piece_idx[j,0],new_piece_idx[j,1]]!=i and current_board[new_piece_idx[j,0],new_piece_idx[j,1]]!= 0: 
                     move_up=False
            else:
                move_up = False # bugs if indexes not in 0,4 range, in this case cnnot allow movement eaither!

            if (0<=new_piece_idx[j+1,0]<=4) and  (0<=new_piece_idx[j+1,1]<=4):
                if current_board[new_piece_idx[j+1,0],new_piece_idx[j+1,1]]!=i and current_board[new_piece_idx[j+1,0],new_piece_idx[j+1,1]]!= 0: 
                     move_down=False
            else:
                move_down = False
            if (0<=new_piece_idx[j+2,0]<=4) and  (0<=new_piece_idx[j+2,1]<=4):
                if current_board[new_piece_idx[j+2,0],new_piece_idx[j+2,1]]!=i and current_board[new_piece_idx[j+2,0],new_piece_idx[j+2,1]]!= 0: 
                     move_left=False
            else:
                move_left = False
            if (0<=new_piece_idx[j+3,0]<=4) and  (0<=new_piece_idx[j+3,1]<=4):
                if current_board[new_piece_idx[j+3,0],new_piece_idx[j+3,1]]!=i and current_board[new_piece_idx[j+3,0],new_piece_idx[j+3,1]]!= 0: 
                     move_right=False
            else:
                move_right= False

        ### à présent on sait si on a le droit de bouger dans les diverses directions, maintenant on génère le déplacement 
        if move_up==True:
            new_positions_tmp = current_board.copy() #.copy important sinon il réécri par dessus!!
            new_positions_tmp[piece_idx[0][:],piece_idx[1][:]] = 0 #met les anciennes positions des blocs à 0 dans la matrice
            new_piece_idx_up = new_piece_idx[np.arange(0,len(new_piece_idx),4)] #on sélectionne les nouvelles coordonnées des blocs après mouvement up
            new_positions_tmp[new_piece_idx_up[:,0], new_piece_idx_up[:,1]] = i #on positionne alors notre piece aux bon endroit
            new_positions.append(new_positions_tmp)
            score_list.append(score)
            move_list.append(f'{i}U')
        if move_down==True:
            new_positions_tmp = current_board.copy() #.copy important sinon il réécri par dessus!!
            new_positions_tmp[piece_idx[0][:],piece_idx[1][:]] = 0 #met les anciennes positions des blocs à 0 dans la matrice
            new_piece_idx_down = new_piece_idx[np.arange(1,len(new_piece_idx),4)] #on sélectionne les nouvelles coordonnées des blocs après mouvement down
            new_positions_tmp[new_piece_idx_down[:,0], new_piece_idx_down[:,1]] = i #on positionne alors notre piece aux bon endroit
            new_positions.append(new_positions_tmp)
            score_list.append(score)
            move_list.append(f'{i}D')
        if move_left==True:
            new_positions_tmp = current_board.copy() #.copy important sinon il réécri par dessus!!
            new_positions_tmp[piece_idx[0][:],piece_idx[1][:]] = 0 #met les anciennes positions des blocs à 0 dans la matrice
            new_piece_idx_left = new_piece_idx[np.arange(2,len(new_piece_idx),4)] #on sélectionne les nouvelles coordonnées des blocs après mouvement left
            new_positions_tmp[new_piece_idx_left[:,0], new_piece_idx_left[:,1]] = i #on positionne alors notre piece aux bon endroit
            new_positions.append(new_positions_tmp)
            score_list.append(score)
            move_list.append(f'{i}L')
        if move_right==True:
            new_positions_tmp = current_board.copy() #.copy important sinon il réécri par dessus!!
            new_positions_tmp[piece_idx[0][:],piece_idx[1][:]] = 0 #met les anciennes positions des blocs à 0 dans la matrice
            new_piece_idx_right = new_piece_idx[np.arange(3,len(new_piece_idx),4)] #on sélectionne les nouvelles coordonnées des blocs après mouvement right
            new_positions_tmp[new_piece_idx_right[:,0], new_piece_idx_right[:,1]] = i #on positionne alors notre piece aux bon endroit
            new_positions.append(new_positions_tmp)
            score_list.append(score)
            move_list.append(f'{i}R')
    new_positions = np.array(new_positions)
    return new_positions, score_list, move_list        


def sort_alongside(array_to_sort, sorter):
    sorted_array = np.zeros_like(array_to_sort)
    for i, idx in enumerate(sorter):
        sorted_array[i] = array_to_sort[idx]
    return sorted_array

# Define the heuristic function (sum of Manhattan distances between initial and final positions of blocks)
def calculate_heuristic(board,final_board):
    heuristic = 0
    for piece in range(0, 10):
        indices = np.where(board == piece)
        final_indices = np.where(final_board == piece)
        heuristic += np.sum(np.abs(indices[0] - final_indices[0]) + np.abs(indices[1] - final_indices[1]))
    return heuristic



final_boards = compute_final_positions()

#on 
initial_board = np.array([[1,2,2,3,4],
                             [1,2,3,3,4],
                             [5,6,7,7,8],
                             [9,9,8,8,8],
                             [0,0,0,0,0]])


current_board = initial_board

# plot_board(initial_positions) #osef c'est board intial

cost_min = 9999999999999
old_cost = 99999999999999999999999999999999999999999999999999999999999999999999999999999999
total_score = 0
total_move = ''
conv= False



final_board = final_boards[7,:,:]

# final_board = np.array([[1,2,2,3,4],
#                         [1,2,3,3,4],
#                         [7,7,0,0,8],
#                         [5,6,8,8,8],
#                         [9,9,0,0,0]])

    
while conv == False: #and total_score<300: #on prie pour que le score n'atteigne pas 300 tbh
    new_boards, score_list,move_list = compute_new_positions(current_board=current_board)
    
    distance_list = np.zeros(len(score_list))
    distance_list2 = []
    distance_list2_keep = []
    
    for i in range(len(score_list)): #on itère à travers tous les moves possibles aka tous les moves de toutes les pièces qui peuvent être faits 
        distance = calculate_heuristic(new_boards[i,:,:], final_board) #calcule le cost minimal entre la new position et la le board final (distance = manhattan)
        distance_list[i] = (distance) #on calcule la distance de tous les next boards vis à vis du final board 
    sorter = np.argsort(distance_list) #calcule comment sort la distance_list pour savoir comment sort tout le monde
    new_boards =  sort_alongside(array_to_sort=new_boards,sorter=sorter) #on sort les new boards 
    score_list = sort_alongside(array_to_sort=score_list,sorter=sorter)
    move_list = np.array([x for _,x in sorted(zip(distance_list,move_list))]) #doit rester en mode liste puisque string ? 
    distance_list = sort_alongside(array_to_sort=distance_list,sorter=sorter) #on sort les distances, on peut alors sélectionner les premières distances et les premiers boards
    
    path_selec = min(5,len(score_list)) #on cherche le min entre le nombre de next boards et 5, 5 étant dle nombre de boards max à explorer  
    
    for j in range(path_selec):
        new_boards2, score_list2, move_list2 = compute_new_positions(current_board=new_boards[j,:,:]) #on calcule les nouveaux boards pour chacun des first boards sélectionnés
        for k in range(len(score_list2)):
            distance2 = calculate_heuristic(new_boards2[k,:,:], final_board)
            distance_list2.append(distance2)
        distance_list2_keep.append(min(distance_list2)) #pour chacun des5 boards sélectionnés, on retient la distance minimale comme ça on sait la dis min de chaque branche et on devra 
        #devra alors dans distance list2 keep  on va chercher l'indice du min ce qui donnera la sélection du du next board parmis les 5 sélectionnés 
            
    
    idx_chosen = np.argmin(distance_list2_keep) #on sélectionne le board avec plus petite distance 
    current_board = new_boards[idx_chosen,:,:]
    total_score += score_list[idx_chosen]
    total_move += move_list[idx_chosen]
    
    # plot_board(initial_positions)
    
    if np.array_equal(current_board,final_board):
        conv = True #si le cost ne varie presque plus, alors on considère que ça a convergé!!
        print(f'The total score is {total_score}')
        print(f'da moves were : {total_move}')


if conv==False:
    print('Fuck')

        
plot_board(current_board) 
plot_board(final_board)