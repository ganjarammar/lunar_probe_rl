import tensorflow as tf
import numpy as np

class LunarLanderAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
    
    def _build_actor(self):
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        # Output 4 nilai untuk 4 thruster, menggunakan sigmoid untuk range 0-1
        outputs = tf.keras.layers.Dense(self.action_dim, activation='sigmoid')(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def _build_critic(self):
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        action_input = tf.keras.layers.Input(shape=(self.action_dim,))
        
        x = tf.keras.layers.Concatenate()([state_input, action_input])
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=[state_input, action_input], outputs=outputs)
    
    def get_action(self, state, noise_scale=0.1):
        state = np.expand_dims(state, axis=0)
        action = self.actor(state).numpy()[0]
        action += np.random.normal(0, noise_scale, size=self.action_dim)
        # Clip ke range [0,1] karena kita menggunakan sigmoid
        return np.clip(action, 0, 1)
    
    def save(self, path):
        self.actor.save_weights(f"{path}/actor")
        self.critic.save_weights(f"{path}/critic")
    
    def load(self, path):
        self.actor.load_weights(f"{path}/actor")
        self.critic.load_weights(f"{path}/critic") 