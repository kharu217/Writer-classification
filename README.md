# Writer Classification
<img src='resrc\hermann_profile.jpg' width="200px" height="250px"> <img src="resrc\goethe_profile.jpg" width="200px" height="250px"><img src="resrc\kafka_profile.jpg" width="200px" height="250px">

# data
__1.2M letters of goethe's books__
> Hermann und Dorothea

__1.5M letters of hermann hesse's books__
> Steppenwolf 

__1.4M letters of kafka's books__
> Before the Law

> An Imperial Message

> Description of a Struggle

# Model

<img src="resrc\model_summary_text.png">

# Train

## Hyper parameters
    Embedding Dimension : 64
    hidden size : 64
    layer number : 32

    Epoch : 150
    Batch size : 32
    Learning rate : 2e-3 ~ 1e-6 (Use LambdaLR schedular)

<img src="resrc\epoch150_.png">

# Result

## Testing with each writer's work

Hermann Hesse - Journey to the East
<img src="resrc\hesse_journey.png">

Goethe - The Sorrows of Young Werther
<img src="resrc\goethe_werther.png">

Kafka - Metamorphosis
<img src="resrc\kafka_metapho.png"> 

