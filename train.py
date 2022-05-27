from src.data.Dataset import Dataset as DS
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from src.models.Model_OneShotRNN import OneShotRNN 


def Train_func(model, **para):
    # Dataset generator
    # n = para['n']
    # m = para['m']
    # k = para['shift']
    # hop = para['hop']
    # input_dim   = para['roll_period'] * n
    # pred_dim    = para['roll_period'] * m
    # shift       = para['roll_period'] * k
    # batch_size  = para['batch_size']
    # normalizer  = para['normalizer']
    # classification = para['classification']
    Data        = DS(**para)
    # model = model(input_dim, pred_dim)
    def compile_model(optimizer, loss, metrics, **kwargs):
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    compile_model(**para)
    # Model compile
    # model.compile(
    #     optimizer=para['optimizer'],
    #     loss=para['loss'],
    #     metrics=para['metrics']
    # )

    # Model fit
    train_history = model.fit(
        Data.Train,
        epochs=para['epochs'],
        verbose=1,
        validation_data=Data.Val,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    # Model evaluate
    test_performance = model.evaluate(Data.Test)

    # Plot examples
    Data.plot_example(50, model=model)

    return train_history ,test_performance, model