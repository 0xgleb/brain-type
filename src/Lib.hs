{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DataKinds             #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds             #-}
{-# LANGUAGE TypeFamilies          #-}
{-# LANGUAGE TypeOperators         #-}

module Lib where

import           Data.Proxy
import           Data.Vector.Sized as VS
import           GHC.TypeLits

data (a :: k) #> b

data Neuron (i :: k) a b = Neuron i a b

data Layer (neurons :: [Neuron k * *]) (a :: *)

data Sum

data Sigmoid

class Summation f i a where
    summation :: Proxy f -> Vector i a -> a

instance Num a => Summation Sum i a where
    summation Proxy = VS.sum

class Activation f a where
    activation :: Proxy f -> a -> a

instance Floating a => Activation Sigmoid a where
    activation Proxy x = 1 / (1 + exp (-x))

data NeuronData i a =
    NeuronData { ndWeights :: Vector i a
               , ndBias    :: a
               } deriving Show

data LayerData i o a =
    LayerData { ldNeurons :: Vector o (NeuronData i a)
              } deriving Show

type family Length xs where
    Length '[]       = 0
    Length (x ': xs) = 1 + (Length xs)

    
class HasData structure where
    type BlockData structure :: *

instance ( Summation sum i a
         , Activation act a
         ) => HasData (Layer '[Neuron sum act] a) where
    type BlockData (Layer '[Neuron i sum act] a) = LayerData i 1 a

instance ( HasData (Layer (Neuron i sum1 act1 ': ns) a)
         , Summation sum i a
         , Activation act a
         ) => HasData (Layer (Neuron i sum act ': Neuron i sum1 act1 ': ns) a) where
    
    type BlockData (Layer (Neuron i sum act ': Neuron i sum1 act1 ': ns) a) =
        LayerData i (Length (Neuron i sum act ': Neuron i sum1 act1 ': ns)) a


class HasData s => Network s i o where
    type Input  s :: *
    type Output s :: *
    runNetwork :: BlockData s -> Input -> Output

instance Network (Layer '[Neuron i sum act] a) where
    type Input  (Layer '[Neuron i sum act] a) = Vector i a
    type Output (Layer '[Neuron i sum act] a) = Vector 1 a
    runNetwork LayerData{..} inputs = VS.singleton $ neuronF $ VS.head ldNeurons
        where neuronF NeuronData{..} =
                activation $ ndBias + summation (VS.zipWith (*) inputs ndWeights)
