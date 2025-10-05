#-----------------------------
#import os
#import numpy as np
#from typing import Tuple, Optional, List, Callable
#
#import tensorflow as tf
#from keras import layers, models, optimizers, callbacks, metrics
#from tensorflow.keras.applications import EfficientNetB0
#from sklearn.model_selection import StratifiedKFold
#from sklearn.utils import class_weight
#
#from Config.config import (
#    IMG_HEIGHT, IMG_WIDTH, CHANNELS, LEARNING_RATE, 
#    MODELS_DIR, PLOTS_DIR, logger, SEED, NUM_CLASSES
#)
#
#INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
#
## -----------------------------
## F1-Score Metric Implementation para múltiplas classes
## -----------------------------
#class F1Score(metrics.Metric):
#    """Implementação personalizada do F1-Score para múltiplas classes"""
#    def _init_(self, name='f1_score', average='weighted', **kwargs):
#        super()._init_(name=name, **kwargs)
#        self.average = average
#        self.precision = metrics.Precision(average=average)
#        self.recall = metrics.Recall(average=average)
#        
#    def update_state(self, y_true, y_pred, sample_weight=None):
#        # Para multiclasse, usamos a classe predita
#        y_pred_classes = tf.argmax(y_pred, axis=1)
#        y_true_classes = tf.argmax(y_true, axis=1) if y_true.shape[-1] > 1 else y_true
#        
#        self.precision.update_state(y_true_classes, y_pred_classes, sample_weight)
#        self.recall.update_state(y_true_classes, y_pred_classes, sample_weight)
#        
#    def result(self):
#        p = self.precision.result()
#        r = self.recall.result()
#        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))
#        
#    def reset_state(self):
#        self.precision.reset_state()
#        self.recall.reset_state()
#        
#    def get_config(self):
#        base_config = super().get_config()
#        return {**base_config, "average": self.average}
#
## -----------------------------
## Data Augmentation
## -----------------------------
#def get_augmentation_layer():
#    """Retorna camada de data augmentation"""
#    return tf.keras.Sequential(
#        [
#            layers.RandomFlip("horizontal"),
#            layers.RandomRotation(0.08),
#            layers.RandomZoom(0.08),
#            layers.RandomTranslation(0.05, 0.05),
#            layers.RandomContrast(0.1),
#        ],
#        name="data_augmentation",
#    )
#
## -----------------------------
## Small CNN para 3 classes
## -----------------------------
#def build_small_cnn(input_shape: Tuple[int,int,int]=INPUT_SHAPE, 
#                   dropout_rate: float = 0.4,
#                   use_augmentation: bool = True,
#                   num_classes: int = NUM_CLASSES):
#    """Arquitetura CNN para 3 classes"""
#    logger.info(f"Construindo small CNN para {num_classes} classes")
#    
#    inputs = layers.Input(shape=input_shape, name="input_image")
#
#    # Data augmentation
#    if use_augmentation:
#        x = get_augmentation_layer()(inputs)
#    else:
#        x = inputs
#
#    # Blocos convolucionais
#    x = layers.Conv2D(32, 3, padding="same")(x)
#    x = layers.BatchNormalization()(x)
#    x = layers.Activation("relu")(x)
#    x = layers.MaxPooling2D(2)(x)
#
#    x = layers.Conv2D(64, 3, padding="same")(x)
#    x = layers.BatchNormalization()(x)
#    x = layers.Activation("relu")(x)
#    x = layers.MaxPooling2D(2)(x)
#    x = layers.Dropout(0.2)(x)
#
#    x = layers.Conv2D(128, 3, padding="same")(x)
#    x = layers.BatchNormalization()(x)
#    x = layers.Activation("relu")(x)
#    x = layers.MaxPooling2D(2)(x)
#    x = layers.Dropout(0.25)(x)
#
#    x = layers.Conv2D(256, 3, padding="same")(x)
#    x = layers.BatchNormalization()(x)
#    x = layers.Activation("relu")(x)
#    x = layers.MaxPooling2D(2)(x)
#    x = layers.Dropout(dropout_rate)(x)
#
#    # Saída para 3 classes
#    x = layers.GlobalAveragePooling2D()(x)
#    x = layers.Dense(128, activation="relu")(x)
#    x = layers.Dropout(dropout_rate)(x)
#    
#    # Camada de saída com softmax para múltiplas classes
#    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
#
#    model = models.Model(inputs, outputs, name="small_cnn_avc_multiclass")
#    logger.info("Small CNN multiclasse criada.")
#    return model
#
## -----------------------------
## Transfer Learning para 3 classes
## -----------------------------
#def build_transfer_model(input_shape: Tuple[int,int,int]=INPUT_SHAPE,
#                         fine_tune: bool = False,
#                         dropout_rate: float = 0.4,
#                         use_augmentation: bool = True,
#                         num_classes: int = NUM_CLASSES):
#    """Modelo de transfer learning para 3 classes"""
#    logger.info(f"Construindo Transfer Learning model para {num_classes} classes")
#    
#    inputs = layers.Input(shape=input_shape, name="input_image")
#    
#    # Data augmentation
#    if use_augmentation:
#        x = get_augmentation_layer()(inputs)
#    else:
#        x = inputs
#
#    # Converter para 3 canais se necessário
#    if input_shape[2] == 1:
#        x = layers.Conv2D(3, (1,1), padding="same", name="to_rgb")(x)
#    
#    # Pré-processamento EfficientNet
#    x = tf.keras.applications.efficientnet.preprocess_input(x)
#
#    # Backbone
#    base_model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet", pooling=None)
#    base_model.trainable = False
#
#    # Cabeça customizada para 3 classes
#    y = base_model.output
#    y = layers.GlobalAveragePooling2D()(y)
#    y = layers.Dropout(dropout_rate)(y)
#    y = layers.Dense(128, activation="relu")(y)
#    y = layers.Dropout(dropout_rate)(y)
#    
#    # Saída para 3 classes
#    outputs = layers.Dense(num_classes, activation="softmax", name="output")(y)
#
#    model = models.Model(inputs, outputs, name="efficientnetb0_avc_multiclass")
#    logger.info("Modelo EfficientNetB0 multiclasse criado.")
#
#    if fine_tune:
#        # Fine-tuning das últimas camadas
#        for layer in base_model.layers[-20:]:
#            if not isinstance(layer, layers.BatchNormalization):
#                layer.trainable = True
#        logger.info("Fine-tuning ativado.")
#
#    return model
#
## -----------------------------
## Compile model para múltiplas classes
## -----------------------------
#def compile_model(model: tf.keras.Model, lr: float = LEARNING_RATE):
#    """Compila o modelo para classificação multiclasse"""
#    logger.info(f"Compilando modelo multiclasse (lr={lr})")
#    
#    opt = optimizers.Adam(learning_rate=lr)
#    model.compile(
#        optimizer=opt,
#        loss="sparse_categorical_crossentropy",  # Para labels inteiros
#        metrics=[
#            metrics.SparseCategoricalAccuracy(name="accuracy"),
#            metrics.AUC(name="auc", multi_label=True),
#            F1Score(name="f1_score"),
#        ],
#    )
#    return model
#
## -----------------------------
## Funções restantes mantidas com pequenos ajustes
## -----------------------------
#def get_common_callbacks(model_dir: str = MODELS_DIR,
#                         plots_dir: str = PLOTS_DIR,
#                         patience_es: int = 7,
#                         reduce_lr_patience: int = 4,
#                         fold: Optional[int] = None):
#    """Callbacks para treinamento"""
#    if fold is not None:
#        model_dir = os.path.join(model_dir, f"fold_{fold}")
#        plots_dir = os.path.join(plots_dir, f"fold_{fold}")
#    
#    os.makedirs(model_dir, exist_ok=True)
#    os.makedirs(plots_dir, exist_ok=True)
#
#    ckpt_path = os.path.join(model_dir, "best_model.keras")
#    csv_path = os.path.join(plots_dir, "training_log.csv")
#    tensorboard_log_dir = os.path.join(plots_dir, "logs")
#
#    cb = [
#        callbacks.ModelCheckpoint(
#            ckpt_path, 
#            monitor="val_f1_score",
#            mode="max",
#            save_best_only=True, 
#            verbose=1
#        ),
#        callbacks.EarlyStopping(
#            monitor="val_f1_score",
#            mode="max",
#            patience=patience_es, 
#            restore_best_weights=True, 
#            verbose=1
#        ),
#        callbacks.ReduceLROnPlateau(
#            monitor="val_f1_score",
#            mode="max",
#            factor=0.5, 
#            patience=reduce_lr_patience, 
#            verbose=1
#        ),
#        callbacks.CSVLogger(csv_path),
#        callbacks.TensorBoard(log_dir=tensorboard_log_dir)
#    ]
#    
#    logger.info("Callbacks preparados para treinamento multiclasse")
#    return cb
#
#def calculate_class_weights(y: np.ndarray) -> dict:
#    """Calcula pesos de classes para dataset desbalanceado"""
#    class_weights = class_weight.compute_class_weight(
#        'balanced',
#        classes=np.unique(y),
#        y=y
#    )
#    return {i: weight for i, weight in enumerate(class_weights)}
#
#def train_with_cross_validation(model_builder: Callable,
#                                X: np.ndarray,
#                                y: np.ndarray,
#                                k_folds: int = 5,
#                                epochs: int = 20,
#                                batch_size: int = 32,
#                                use_class_weights: bool = True):
#    """Treinamento com validação cruzada para múltiplas classes"""
#    logger.info(f"Iniciando K-Fold estratificado com k={k_folds}")
#    
#    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=SEED)
#    histories = []
#    model_paths = []
#    
#    if use_class_weights:
#        class_weights = calculate_class_weights(y)
#        logger.info(f"Pesos de classe calculados: {class_weights}")
#    else:
#        class_weights = None
#
#    fold = 0
#    for train_idx, val_idx in skf.split(X, y):
#        fold += 1
#        logger.info(f"Fold {fold}/{k_folds} - Treinando modelo")
#        
#        X_train, X_val = X[train_idx], X[val_idx]
#        y_train, y_val = y[train_idx], y[val_idx]
#
#        model = model_builder()
#        model = compile_model(model)
#
#        callbacks_list = get_common_callbacks(fold=fold)
#
#        history = model.fit(
#            X_train, y_train,
#            validation_data=(X_val, y_val),
#            epochs=epochs,
#            batch_size=batch_size,
#            callbacks=callbacks_list,
#            verbose=1,
#            class_weight=class_weights
#        )
#        
#        histories.append(history)
#        out_path = os.path.join(MODELS_DIR, f"model_fold_{fold}.keras")
#        model.save(out_path)
#        model_paths.append(out_path)
#        logger.info(f"Fold {fold} concluído, modelo salvo em {out_path}")
#
#    return histories, model_paths
#
#def predict_with_ensemble(model_paths: List[str], 
#                         X: np.ndarray) -> np.ndarray:
#    """Predição com ensemble de modelos"""
#    predictions = []
#    
#    for model_path in model_paths:
#        model = tf.keras.models.load_model(
#            model_path, 
#            custom_objects={'F1Score': F1Score}
#        )
#        pred = model.predict(X)
#        predictions.append(pred)
#    
#    # Média das predições
#    avg_predictions = np.mean(predictions, axis=0)
#    
#    # Classe predita
#    class_predictions = np.argmax(avg_predictions, axis=1)
#    
#    return class_predictions, avg_predictions
#
#def save_model(model: tf.keras.Model, filename: str = "modelo_avc_multiclass.keras"):
#    """Salva modelo treinado"""
#    os.makedirs(MODELS_DIR, exist_ok=True)
#    out_path = os.path.join(MODELS_DIR, filename)
#    model.save(out_path, save_format='keras')
#    logger.info(f"Modelo salvo em: {out_path}")
#    return out_path
#
#def build_model(model_type: str = 'small_cnn', **kwargs):
#    """Função utilitária para selecionar tipo de modelo"""
#    if model_type == 'small_cnn':
#        return build_small_cnn(**kwargs)
#    elif model_type == 'transfer_learning':
#        return build_transfer_model(fine_tune=False, **kwargs)
#    elif model_type == 'transfer_learning_finetune':
#        return build_transfer_model(fine_tune=True, **kwargs)
#    else:
#        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")
#