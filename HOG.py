import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def get_differential_filter():
    #sobel filter
    filter_x = [[1,0,-1],[2,0,-2],[1,0,-1]]
    filter_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
    return filter_x, filter_y

def filter_image(im, filter):
    # pad zero around 
    im_pad0 =np.zeros((im.shape[0]+2,im.shape[1]+2))
    for i in range(1,im_pad0.shape[0]-1):
        for j in range(1,im_pad0.shape[1]-1):
            im_pad0[i][j]=im[i-1][j-1]
    #filter image
    im_filtered0 = np.zeros((im.shape[0]+2,im.shape[1]+2))
    im_filtered = np.zeros((im.shape[0],im.shape[1]))
    #filter = [[1,0,-1],[1,0,-1],[1,0,-1]]
    for i in range(1,im_pad0.shape[0]-1):
        for j in range(1,im_pad0.shape[1]-1):
            #calculate each pixel
            im_filtered0[i][j]=im_pad0[i-1][j-1]*filter[0][0]+im_pad0[i-1][j]*filter[0][1]+im_pad0[i-1][j+1]*filter[0][2] \
            +im_pad0[i][j-1]*filter[1][0]+im_pad0[i][j]*filter[1][1]+im_pad0[i][j+1]*filter[1][2] \
            +im_pad0[i+1][j-1]*filter[2][0]+im_pad0[i+1][j]*filter[2][1]+im_pad0[i+1][j+1]*filter[2][2]
            #this method is from the slide
            #v=0
            #for k in range(0,3):
            #    for l in range (0,3):
            #        i1 = i+k-1
            #        j1 = j+l-1
            #        v = v+im_pad0[i1][j1]*filter[k][l]
            #im_filtered0[i][j]=v
            
    #get rid of surrounding 0
    for i in range(0,im.shape[0]):
        for j in range(0,im.shape[1]):
            im_filtered[i][j]=im_filtered0[i+1][j+1]
            
    plt.imshow(im_filtered)
    
    return im_filtered


def get_gradient(im_dx, im_dy):
    #grad_magnitude
    grad_mag=np.zeros((im_dx.shape[0],im_dx.shape[1]))
    for i in range(0,im_dx.shape[0]):
        for j in range(0,im_dx.shape[1]):
            grad_mag[i][j]=math.sqrt(im_dx[i][j]*im_dx[i][j]+im_dy[i][j]*im_dy[i][j])
            
    plt.imshow(grad_mag)
    
    #grad_angle
    grad_angle=np.zeros((im_dx.shape[0],im_dx.shape[1]))
    for i in range(0,im_dx.shape[0]):
        for j in range(0,im_dx.shape[1]):
            grad_angle[i][j]= np.arctan(im_dy[i][j]/im_dx[i][j])
            #restrict angle to [0,pi)
            if grad_angle[i][j]<0 :
                grad_angle[i][j]=grad_angle[i][j]+math.pi
    plt.imshow(grad_angle)
    
    
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    #ori_histo size, ignore the undivisible area
    ori_histo = np.zeros((int(grad_mag.shape[0]/cell_size),int(grad_mag.shape[1]/cell_size),6))
    #calculate per cell
    for i in range(0,int(grad_mag.shape[0]/cell_size)):
        for j in range(0,int(grad_mag.shape[1]/cell_size)):
            #ori_histo[i][j] per cell
            for k in range(0,cell_size):
                for l in range(0,cell_size):
                    #per pixel [i*cell_size+k][j*cell_size+l]
                    pixel_angle = grad_angle[i*cell_size+k][j*cell_size+l]
                    if (pixel_angle>=math.pi*(165/180) and pixel_angle<math.pi*(180/180)) or (pixel_angle>=0 and pixel_angle<math.pi*(15/180)):
                        ori_histo[i][j][0]=ori_histo[i][j][0]+grad_mag[i*cell_size+k][j*cell_size+l]
                    if pixel_angle>=math.pi*(15/180) and pixel_angle<math.pi*(45/180):
                        ori_histo[i][j][1]=ori_histo[i][j][1]+grad_mag[i*cell_size+k][j*cell_size+l]
                    if pixel_angle>=math.pi*(45/180) and pixel_angle<math.pi*(75/180):
                        ori_histo[i][j][2]=ori_histo[i][j][2]+grad_mag[i*cell_size+k][j*cell_size+l]
                    if pixel_angle>=math.pi*(75/180) and pixel_angle<math.pi*(105/180):
                        ori_histo[i][j][3]=ori_histo[i][j][3]+grad_mag[i*cell_size+k][j*cell_size+l]
                    if pixel_angle>=math.pi*(105/180) and pixel_angle<math.pi*(135/180):
                        ori_histo[i][j][4]=ori_histo[i][j][4]+grad_mag[i*cell_size+k][j*cell_size+l]
                    if pixel_angle>=math.pi*(135/180) and pixel_angle<math.pi*(165/180):
                        ori_histo[i][j][5]=ori_histo[i][j][5]+grad_mag[i*cell_size+k][j*cell_size+l]
    
    
    return ori_histo


def get_block_descriptor(ori_histo, block_size):

    ori_histo_normalized = np.zeros((ori_histo.shape[0]-(block_size-1),ori_histo.shape[1]-(block_size-1),6*block_size*block_size))
    #
    e=0.001
    for i in range(0,ori_histo.shape[0]-1):
        for j in range(0,ori_histo.shape[1]-1):
            #ori_histo_normalized[i][j]
            square_sum = e*e
            hi_list=[]
            for x in range(0,block_size):
                for y in range(0,block_size):
                    for z in range(0,6):
                        hi_list.append(ori_histo[i+x][j+y][z])
                        square_sum = square_sum+(ori_histo[i+x][j+y][z]*ori_histo[i+x][j+y][z])
            divisor = math.sqrt(square_sum)
              
            for m in range(0,6*block_size*block_size):
                ori_histo_normalized[i][j][m]=hi_list[m]/divisor

            
    return ori_histo_normalized

def extract_hog(im):
    # convert grey-scale image to double format
    im = im.astype('float') / 255.0    

    (filter_x, filter_y) = get_differential_filter()
    im_dx = filter_image(im, filter_x)
    im_dy = filter_image(im, filter_y)
    (grad_mag, grad_angle) = get_gradient(im_dx, im_dy)
    cell_size = 8
    ori_histo=build_histogram(grad_mag, grad_angle, cell_size)
    #visualize_hog(im, ori_histo, 8)
    visualize_hog_cell(im, ori_histo, cell_size)
    block_size = 2
    ori_histo_normalized=get_block_descriptor(ori_histo, block_size)
    
    (x,y,z)=ori_histo_normalized.shape
    hog=[]
    for i in range(0,x):
        for j in range(0,y):
            for k in range(0,z):
                hog.append(ori_histo_normalized[i][j][k])
    hog = np.asarray(hog)
    visualize_hog_block(im, hog, cell_size, block_size)
    return hog


def visualize_hog(im, ori_histo, cell_size):
    norm_constant = 1e-3
    num_bins = ori_histo.shape[2]
    height, width = im.shape
    max_len = cell_size / 3
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size/2:width:cell_size], np.r_[cell_size/2:height:cell_size])

    bin_ave = np.sqrt(np.sum(ori_histo ** 2, axis=2) + norm_constant ** 2)  # (ori_histo.shape[0], ori_histo.shape[1])
    histo_normalized = ori_histo / np.expand_dims(bin_ave, axis=2) * max_len  # same dims as ori_histo
    mesh_u = histo_normalized * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - mesh_u[:, :, i], mesh_y - mesh_v[:, :, i], 2 * mesh_u[:, :, i], 2 * mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()
    plt.savefig("histogram.png", bbox_inches='tight')
    
def visualize_hog_cell(im, ori_histo, cell_size):
    
    norm_constant = 1e-3
    num_cell_h, num_cell_w, num_bins = ori_histo.shape
    max_len = cell_size / 3
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size/2: cell_size*num_cell_w: cell_size], np.r_[cell_size/2: cell_size*num_cell_h: cell_size])
    bin_ave = np.sqrt(np.sum(ori_histo ** 2, axis=2) + norm_constant ** 2)  # (ori_histo.shape[0], ori_histo.shape[1])
    histo_normalized = ori_histo / np.expand_dims(bin_ave, axis=2) * max_len  # same dims as ori_histo
    mesh_u = histo_normalized * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - mesh_u[:, :, i], mesh_y - mesh_v[:, :, i], 2 * mesh_u[:, :, i], 2 * mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()
    
# visualize histogram of each block
def visualize_hog_block(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7 # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[int(cell_size*block_size/2): cell_size*num_cell_w-(cell_size*block_size/2)+1: cell_size], np.r_[int(cell_size*block_size/2): cell_size*num_cell_h-(cell_size*block_size/2)+1: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins)) # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='white', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    plt.show()

def visualize_color(image):
    Vmax=0.0
    Vmin=0.0
    
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            if image[i][j] > Vmax:
                Vmax=image[i][j]
            if image[i][j] < Vmin:
                Vmin = image[i][j]
    
    image_color = np.zeros((image.shape[0], image.shape[1], 1), dtype = "uint8")
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            image_color[i][j][0]=int(255*((image[i][j]-Vmin)/(Vmax-Vmin)))
            
    image_color = cv2.applyColorMap(image_color, cv2.COLORMAP_BONE)
    cv2.imshow("colored",image_color)
    cv2.waitKey()



if __name__=='__main__':
    im = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    #im = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
    #plt.imshow(im)
    hog = extract_hog(im)
    
    #visualize_color(im_dx)
    

