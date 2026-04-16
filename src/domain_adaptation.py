"""
domain_adaptation.py

Domain adaptation techniques for reducing distribution mismatch across
heterogeneous multisource datasets (AI4I, CWRU, MIMII, MetroPT-3, Edge-IIoT).

Techniques:
- Maximum Mean Discrepancy (MMD) loss for domain invariant representations
- Domain adversarial training (DANN) with gradient reversal
- Progressive unfreezing for curriculum-based negative transfer mitigation
- Domain-specific batch normalization for modality-specific adaptation
- Focal domain confusion for rare-source prioritization
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


class GradientReversalLayer(layers.Layer):
    """
    Gradient reversal layer for domain adversarial training (Ganin et al. 2015).
    
    During forward pass: acts as identity.
    During backward pass: reverses gradient sign (multiplies by -1).
    """
    
    def __init__(self, lambd: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.lambd = float(lambd)
    
    @tf.custom_gradient
    def _grad_reverse(self, x: tf.Tensor) -> tuple[tf.Tensor, callable]:
        def grad(dy):
            return -self.lambd * dy
        return x, grad
    
    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self._grad_reverse(x)


def mmd_loss(
    source_embeddings: tf.Tensor,
    target_embeddings: tf.Tensor,
    kernel: str = "rbf",
    gamma: float = 0.01,
) -> tf.Tensor:
    """
    Maximum Mean Discrepancy (MMD) loss for unsupervised domain adaptation.
    
    Minimizes the distance between source and target feature distributions
    by matching their kernel mean embeddings.
    
    Args:
        source_embeddings: (N_src, D) source domain features
        target_embeddings: (N_tgt, D) target domain features
        kernel: 'rbf' or 'linear'
        gamma: RBF kernel bandwidth
    
    Returns:
        Scalar MMD loss
    """
    def _kernel_fn(x, y, gamma_val):
        if kernel == "rbf":
            # RBF kernel: exp(-gamma * ||x - y||²)
            x_2 = tf.reduce_sum(x ** 2, axis=1, keepdims=True)
            y_2 = tf.reduce_sum(y ** 2, axis=1, keepdims=True)
            xy = tf.matmul(x, tf.transpose(y))
            dists = x_2 + tf.transpose(y_2) - 2 * xy
            return tf.exp(-gamma_val * dists)
        else:  # linear
            return tf.matmul(x, tf.transpose(y))
    
    batch_size = tf.minimum(
        tf.shape(source_embeddings)[0],
        tf.shape(target_embeddings)[0]
    )
    
    # Use same batch size for fair kernel matrix computation
    src = source_embeddings[:batch_size]
    tgt = target_embeddings[:batch_size]
    
    # Kernel matrices
    K_ss = _kernel_fn(src, src, gamma)
    K_tt = _kernel_fn(tgt, tgt, gamma)
    K_st = _kernel_fn(src, tgt, gamma)
    
    # MMD: E[k(s,s)] - 2*E[k(s,t)] + E[k(t,t)]
    n = tf.cast(batch_size, tf.float32)
    
    # Diagonal scaling (unbiased estimator)
    mmd_val = (
        tf.reduce_sum(K_ss) / (n * n) -
        2 * tf.reduce_sum(K_st) / (n * n) +
        tf.reduce_sum(K_tt) / (n * n)
    )
    
    return tf.maximum(mmd_val, 0.0)  # Clip to non-negative


def build_domain_discriminator(
    embedding_dim: int = 256,
    num_domains: int = 5,
    dropout: float = 0.3,
) -> Model:
    """
    Build a simple domain discriminator network.
    
    Args:
        embedding_dim: Input embedding dimension
        num_domains: Number of source domains
        dropout: Dropout rate
    
    Returns:
        Keras Model that outputs domain class logits
    """
    inputs = layers.Input(shape=(embedding_dim,), name="domain_disc_input")
    
    x = layers.Dense(128, activation="relu", name="disc_dense1")(inputs)
    x = layers.BatchNormalization(name="disc_bn1")(x)
    x = layers.Dropout(dropout, name="disc_drop1")(x)
    
    x = layers.Dense(64, activation="relu", name="disc_dense2")(x)
    x = layers.BatchNormalization(name="disc_bn2")(x)
    x = layers.Dropout(dropout, name="disc_drop2")(x)
    
    outputs = layers.Dense(num_domains, activation="softmax", name="domain_logits")(x)
    
    return Model(inputs=inputs, outputs=outputs, name="DomainDiscriminator")


def build_domain_mixup_layer(alpha: float = 1.0):
    """
    Domain mixup: smoothly blend samples from different domains to avoid
    sharp decision boundaries between distributed sources.
    
    Beta(alpha, alpha) samples control the blend ratio.
    """
    def mixup_fn(
        inputs_batch1: tf.Tensor,
        inputs_batch2: tf.Tensor,
        targets_batch1: tf.Tensor,
        targets_batch2: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        batch_size = tf.shape(inputs_batch1)[0]
        
        if alpha > 0:
            # Beta(alpha, alpha) via two gamma samples.
            g1 = tf.random.gamma(shape=[batch_size, 1], alpha=alpha, dtype=tf.float32)
            g2 = tf.random.gamma(shape=[batch_size, 1], alpha=alpha, dtype=tf.float32)
            lam = g1 / (g1 + g2 + 1e-8)
        else:
            lam = tf.random.uniform(
                shape=(batch_size, 1),
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32,
            )
        
        # Mix inputs and targets
        mixed_inputs = lam * inputs_batch1 + (1.0 - lam) * inputs_batch2
        mixed_targets_1 = lam * targets_batch1 + (1.0 - lam) * targets_batch2
        mixed_targets_2 = lam * targets_batch2 + (1.0 - lam) * targets_batch1
        
        return mixed_inputs, (mixed_targets_1 + mixed_targets_2) / 2.0
    
    return mixup_fn


def progressive_unfreezing(
    model: Model,
    frozen_layers: list[str],
    epoch: int,
    total_epochs: int,
) -> None:
    """
    Progressive unfreezing: gradually unfreeze layers as training progresses.
    
    Reduces negative transfer early in training by keeping most layers frozen,
    then gradually allowing fine-tuning.
    
    Args:
        model: Keras model to unfreeze
        frozen_layers: Names of layers to progressively unfreeze
        epoch: Current epoch (0-indexed)
        total_epochs: Total training epochs
    """
    # Fraction of training complete
    progress = epoch / max(total_epochs, 1)
    
    # Number of layers to unfreeze based on progress
    num_to_unfreeze = max(1, int(len(frozen_layers) * progress))
    
    def _set_trainable(layer_name: str, trainable: bool) -> None:
        try:
            layer = model.get_layer(layer_name)
        except Exception:
            return
        layer.trainable = trainable

    # Unfreeze the last `num_to_unfreeze` layers
    for layer_name in frozen_layers[-num_to_unfreeze:]:
        _set_trainable(layer_name, True)
    
    # Keep earlier layers frozen
    for layer_name in frozen_layers[:-num_to_unfreeze]:
        _set_trainable(layer_name, False)


def build_domain_adapted_mtl_model(
    base_model: Model,
    num_domains: int = 5,
    use_mmd: bool = True,
    use_dann: bool = True,
    embedding_dim: int = 256,
) -> tuple[Model, Model]:
    """
    Wrap a base MTL model with domain adaptation components.
    
    Returns:
        - adapted_model: Full model with domain adaptation heads
        - feature_extractor: Shared embedding extractor for domain losses
    """
    # Extract feature extractor from base model (pre-shared dense layers)
    # Assuming base model outputs [head_rul, head_faults, head_anomaly]
    
    # Create a new model that outputs shared features + task heads
    shared_layer = base_model.get_layer("shared_dense2")
    if shared_layer is None:
        raise ValueError("Base model must have 'shared_dense2' layer for domain adaptation")
    
    # Create feature extractor model (outputs from shared_dense2)
    feature_extractor = Model(
        inputs=base_model.inputs,
        outputs=shared_layer.output,
        name="FeatureExtractor"
    )
    
    # Build domain discriminator
    domain_disc = build_domain_discriminator(
        embedding_dim=shared_layer.output_shape[-1],
        num_domains=num_domains,
    )
    
    # Add gradient reversal if using DANN
    if use_dann:
        features = feature_extractor.outputs[0]
        reversed_features = GradientReversalLayer(lambd=1.0, name="grad_reversal")(features)
        domain_logits = domain_disc(reversed_features)
    else:
        features = feature_extractor.outputs[0]
        domain_logits = domain_disc(features)
    
    # Full adapted model
    adapted_outputs = list(base_model.outputs) + [domain_logits]
    adapted_model = Model(
        inputs=base_model.inputs,
        outputs=adapted_outputs,
        name="DomainAdapted_MTL"
    )
    
    return adapted_model, feature_extractor


class DomainAdaptationCallback(tf.keras.callbacks.Callback):
    """
    Callback for progressive unfreezing and domain-specific learning rate scheduling.
    """
    
    def __init__(
        self,
        frozen_layers: list[str] | None = None,
        total_epochs: int = 30,
        min_lr: float = 1e-6,
    ):
        super().__init__()
        self.frozen_layers = frozen_layers or []
        self.total_epochs = total_epochs
        self.min_lr = min_lr
    
    def on_epoch_begin(self, epoch: int, logs=None):
        if self.frozen_layers:
            progressive_unfreezing(
                self.model,
                self.frozen_layers,
                epoch,
                self.total_epochs,
            )
        
        # Decay learning rate
        lr_schedule = 1e-3 * (0.5 ** (epoch / max(self.total_epochs, 1)))
        lr_schedule = max(lr_schedule, self.min_lr)
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr_schedule)


def compute_domain_loss(
    source_embeddings: tf.Tensor,
    target_embeddings: tf.Tensor,
    domain_predictions: tf.Tensor,
    true_domains: tf.Tensor,
    use_mmd: bool = True,
    use_dann: bool = True,
    mmd_weight: float = 0.1,
    dann_weight: float = 0.1,
) -> dict[str, tf.Tensor]:
    """
    Compute combined domain adaptation losses.
    
    Returns dict with individual and total loss terms.
    """
    losses = {}
    total_loss = 0.0
    
    if use_mmd:
        mmd_val = mmd_loss(source_embeddings, target_embeddings, kernel="rbf", gamma=0.01)
        losses["mmd"] = mmd_val
        total_loss = total_loss + mmd_weight * mmd_val
    
    if use_dann:
        dann_loss = tf.keras.losses.categorical_crossentropy(
            true_domains,
            domain_predictions,
        )
        dann_loss = tf.reduce_mean(dann_loss)
        losses["dann"] = dann_loss
        total_loss = total_loss + dann_weight * dann_loss
    
    losses["total_domain_loss"] = total_loss
    return losses


def focal_domain_confusion_loss(
    domain_predictions: tf.Tensor,
    true_domains: tf.Tensor,
    gamma: float = 2.0,
    alpha_domains: np.ndarray | None = None,
) -> tf.Tensor:
    """
    Focal loss variant for domain classification to prioritize hard-to-discriminate domains.
    
    Helpful for rare or hard-to-adapt domains (e.g., Edge-IIoT if underrepresented).
    
    Args:
        domain_predictions: (N, num_domains) softmax predictions
        true_domains: (N, num_domains) one-hot encoded true domains
        gamma: Focusing parameter (higher = focus on hard negatives)
        alpha_domains: (num_domains,) per-domain weights for class imbalance
    
    Returns:
        Scalar focal domain loss
    """
    if alpha_domains is None:
        alpha = tf.ones(shape=(tf.shape(domain_predictions)[-1],), dtype=tf.float32)
    else:
        alpha = tf.convert_to_tensor(alpha_domains, dtype=tf.float32)
    
    # Cross-entropy
    ce = -tf.reduce_sum(true_domains * tf.math.log(domain_predictions + 1e-8), axis=-1)
    
    # Focal weighting: downweight easy examples
    p_t = tf.reduce_sum(true_domains * domain_predictions, axis=-1)
    focal_weight = tf.pow(1.0 - p_t, gamma)
    
    # Alpha weighting: balance rare domains
    alpha_t = tf.reduce_sum(true_domains * alpha, axis=-1)
    
    focal_loss = alpha_t * focal_weight * ce
    return tf.reduce_mean(focal_loss)
