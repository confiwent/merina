import os
import torch
import numpy as np


def save_models(logging, summary_dir, add_str, model_actor, vae_net, epoch,
                max_QoE, mean_value):
    save_path = summary_dir + '/' + add_str + '/models'
    if not os.path.exists(save_path): os.system('mkdir ' + save_path)
    
    save_flag = False
    if len(max_QoE) < 5: # save five models that perform best
        save_flag = True
        max_QoE[epoch] = mean_value
    elif mean_value > min(max_QoE.values()):
        min_idx = 0
        for key, value in max_QoE.items(): ## find the model that should be remove 
            if value == min(max_QoE.values()):
                min_idx = key if key > min_idx else min_idx

        if min_idx > 0:
            actor_remove_path = summary_dir + '/' + add_str + '/' + 'models' + \
                            "/%s_%s_%d.model" %(str('policy'), add_str, int(min_idx))
            vae_remove_path = summary_dir + '/' + add_str + '/' + 'models' + \
                            "/%s_%s_%d.model" %(str('VAE'), add_str, int(min_idx))
            if os.path.exists(actor_remove_path): os.system('rm ' + actor_remove_path)
            # if os.path.exists(critic_save_path): os.system('rm ' + critic_save_path)
            if os.path.exists(vae_remove_path): os.system('rm ' + vae_remove_path)

        save_flag = True
        max_QoE.pop(min_idx)
        max_QoE[epoch] = mean_value
        
    if save_flag:
        logging.info("Model saved in file")
        # save models
        actor_save_path = summary_dir + '/' + add_str + '/' + 'models' + \
                            "/%s_%s_%d.model" %(str('policy'), add_str, int(epoch))
        # critic_save_path = summary_dir + '/' + add_str + "/%s_%s_%d.model" %(str('critic'), add_str, int(epoch))
        vae_save_path = summary_dir + '/' + add_str + '/' + 'models' + \
                            "/%s_%s_%d.model" %(str('VAE'), add_str, int(epoch))
        if os.path.exists(actor_save_path): os.system('rm ' + actor_save_path)
        # if os.path.exists(critic_save_path): os.system('rm ' + critic_save_path)
        if os.path.exists(vae_save_path): os.system('rm ' + vae_save_path)
        torch.save(model_actor.state_dict(), actor_save_path)
        # torch.save(model_critic.state_dict(), critic_save_path)
        torch.save(vae_net.state_dict(), vae_save_path)