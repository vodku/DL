import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def ploting_images(paths, ncols, nrows):
    #-----------------------------------------------------
    # setting up the subplot
    #-----------------------------------------------------
    ncols = ncols
    nrows = nrows
    fig = plt.gcf()
    fig.set_size_inches(ncols*4, nrows*4)
    images = []
    
    for path in paths:
        pic_index = np.random.randint(0, (len(os.listdir(path))-(ncols*nrows)))
        images.extend([os.path.join(path, fname) for fname in os.listdir(path)[pic_index-round((ncols*nrows)/2):pic_index]])

    for i, img_path in enumerate(images):
        sp = plt.subplot(nrows, ncols, i+1)
        img = mpimg.imread(img_path)
        plt.imshow(img)
        
    plt.show()
    

def plot_accuracy_loss(history):
    #-----------------------------------------------------
    # Retrieve the results on training and validation data
    #-----------------------------------------------------
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    #-----------------------------------------------------
    # Plot training and validation accuracy per epoch
    #-----------------------------------------------------
    ax1 = plt.subplot(222)
    sns.lineplot(x=epochs, y=acc, ax=ax1)
    sns.lineplot(x=epochs, y=val_acc, ax=ax1)
    ax1.set_title('Training and validation accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy [%]')

    #-----------------------------------------------------
    # Plot training and validation loss per epoch
    #-----------------------------------------------------
    ax2 = plt.subplot(221)
    sns.lineplot(x=epochs, y=loss, ax=ax2)
    sns.lineplot(x=epochs, y=val_loss, ax=ax2)
    ax2.set_title('Training and validation loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    plt.show()
    
def visualize_features_output(model, array_files):    
    all_list = []
    #-----------------------------------------------------
    # Get the outputs of the layers of the model
    # Redefine model to output right after the first hidden layer
    #-----------------------------------------------------
    successive_outputs = [layer.output for layer in model.layers[1:]]

    #-----------------------------------------------------
    # Build a new model with the functional API
    #-----------------------------------------------------
    visualization_model = Model(inputs=model.input, outputs= successive_outputs)

    #-----------------------------------------------------
    # Randomly choose an image from cats and dogs
    # from the training sets
    #-----------------------------------------------------
    for array in array_files:
        all_list.extend(array)
    img_path = random.choice(all_list)
    img = load_img(img_path, target_size=(150, 150))

    #-----------------------------------------------------
    # Preprocessing
    #-----------------------------------------------------
    img_array = img_to_array(img)
    img_array = img_array.reshape((1,) + img_array.shape)
    img_array /= 255.0

    #-----------------------------------------------------
    # Prediction
    #-----------------------------------------------------
    successive_feature_maps = visualization_model.predict(img_array)

    #-----------------------------------------------------
    # Get the layer's names
    #-----------------------------------------------------
    layer_names = [layer.name for layer in model.layers[1:]]

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            n_features = feature_map.shape[-1]
            size = feature_map.shape[1]
            display_grid = np.zeros((size, size*n_features))

            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x = normalize(x, None, alpha = 0, beta = 255, norm_type = NORM_MINMAX, dtype = CV_32F)
                x = x.astype(np.uint8)
                display_grid[:, i*size:(i+1)*size] = x

            scale = 40./n_features
            plt.figure(figsize=(scale*n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
