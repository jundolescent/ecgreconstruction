from model.lstm import ECGReconstructor
from model.autoencoder import ECGAutoencoder
from utils.argument import parse_arguments

# data_loader = ECGDataLoader(path='./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/', sampling_rate=100)
# X_train, y_train, X_test, y_test = data_loader.preprocess_data(test_fold=10)

args = parse_arguments()

if args.model == 'lstm':
    # LSTM
    reconstructor = ECGReconstructor(path='./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/')
    reconstructor.train()
    reconstructor.test()

elif args.model == 'autoencoder':
    # Autoencoder
    autoencoder = ECGAutoencoder(input_size=1000, encoding_dim=128, learning_rate=0.001, batch_size=64, num_epochs=10, path='./ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/')
    autoencoder.train()
    autoencoder.test()
    reconstructed_data = autoencoder.reconstruct()


