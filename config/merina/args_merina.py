import argparse

def get_args(rest_args):
    parser = argparse.ArgumentParser(description='MeRL-based ABR')
    parser.add_argument('--test', action='store_true', help='Evaluate only')


    parser.add_argument('--log', action='store_true', help='Use logarithmic form QoE metric')
    parser.add_argument('--adp', action='store_true', help='adaptation')
    parser.add_argument('--non-il', action='store_true', help='Not use imitation learning for pre-training')
    parser.add_argument('--name', default='merina', help='the name of result folder')


    parser.add_argument('--mpc-h', nargs='?', const=5, default=5, type=int, help='The MPC planning horizon')
    parser.add_argument('--valid-i',nargs='?', const=100, default=100, type=int, help='The valid interval')
    
    ## Latent encoder 
    parser.add_argument('--gru', action='store_true', help='Use GRU encoder')
    parser.add_argument('--latent-dim', nargs='?', const=16, default=16, type=int, help='The dimension of latent space')
    parser.add_argument('--kld-beta', nargs='?', const=0.01, default=0.01, type=float, help='The coefficient of kld in the VAE loss function')
    parser.add_argument('--kld-lambda', nargs='?', const=1.1, default=1.1, type=float, help='The coefficient of kld in the VAE recon loss function') ## control the strength of over-fitting of reconstruction, KL divergence between the prior P(D) and the distribution of P(D|\theta)
    parser.add_argument('--vae-gamma', nargs='?', const=0.7, default=0.7, type=float, help='The coefficient of reconstruction loss in the VAE loss function')
    
    ## Policy loss 
    parser.add_argument('--lc-alpha', nargs='?', const=1, default=1, type=float, help='The coefficient of cross entropy in the actor loss function')
    parser.add_argument('--lc-beta', nargs='?', const=0.25, default=0.25, type=float, help='The coefficient of entropy in the imitation loss function')
    parser.add_argument('--lc-mu', nargs='?', const=0.1, default=0.1, type=float, help='The coefficient of cross entropy in the actor loss function')
    parser.add_argument('--lc-gamma', nargs='?', const=0.20, default=0.20, type=float, help='The coefficient of mutual information in the actor loss function')
    parser.add_argument('--sp-n', nargs='?', const=10, default=10, type=int, help='The sample numbers of the mutual information')
    parser.add_argument('--gae-gamma', nargs='?', const=0.99, default=0.99, type=float, help='The gamma coefficent for GAE estimation')
    parser.add_argument('--gae-lambda', nargs='?', const=0.95, default=0.95, type=float, help='The lambda coefficent for GAE estimation')
    
    ## PPO configures 
    parser.add_argument('--batch-size', nargs='?', const=64, default=64, type=int, help='Minibatch size for training')
    parser.add_argument('--ppo-ups', nargs='?', const=2, default=2, type=int, help='Update numbers in each epoch for PPO')
    parser.add_argument('--explo-num', nargs='?', const=32, default=32, type=int, help='Exploration steps for roll-out')
    parser.add_argument('--ro-len', nargs='?', const=25, default=25, type=int, help='Length of roll-out')
    parser.add_argument('--clip', nargs='?', const=0.01, default=0.01, type=float, help='Clip value of ppo')
    parser.add_argument('--anneal-p', nargs='?', const=0.95, default=0.95, type=float, help='Annealing parameters for entropy regularization')
    
    
    ## choose datasets for throughput traces 
    parser.add_argument('--tf', action='store_true', help='Use FCC traces')
    parser.add_argument('--tfh', action='store_true', help='Use FCC_and_3GP traces')
    parser.add_argument('--to', action='store_true', help='Use Oboe traces')
    parser.add_argument('--t3g', action='store_true', help='Use 3GP traces')
    parser.add_argument('--tp', action='store_true', help='Use Puffer traces')
    parser.add_argument('--tp2', action='store_true', help='Use Puffer2 traces')

    return parser.parse_args(rest_args)