{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleInstances #-}

module Main where

import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector as V
import System.Random
import Control.Monad (foldM, when, forever)
import Data.List (foldl')
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy.Char8 as BSL
import qualified Data.Binary as Binary
import Data.Binary.Get
import System.IO
import qualified Data.Aeson as Aeson
import Data.Aeson (ToJSON(..), FromJSON(..), object, (.=))
import GHC.Generics
import Control.DeepSeq
import System.Directory (doesFileExist)
import Text.Printf
import Control.Concurrent
import System.Environment (getArgs)
import Data.String (fromString)
import qualified Data.Vector.Generic as VG


type Weight = Double
type Bias = Double
type Activation = Double
type Delta = Double
type LearningRate = Double

data Neuron = Neuron {
    weights :: !(VU.Vector Weight),
    bias :: !Bias
} deriving (Show, Generic)

instance NFData Neuron

data Layer = Layer {
    neurons :: !(V.Vector Neuron),
    activations :: !(VU.Vector Activation)
} deriving (Show, Generic)

instance NFData Layer

data ConvFilter = ConvFilter {
    filterWeights :: !(V.Vector (VU.Vector Weight)),
    filterBias :: !Bias
} deriving (Show, Generic)

instance NFData ConvFilter

data CNN = CNN {
    convLayer1 :: !(V.Vector ConvFilter),
    convLayer2 :: !(V.Vector ConvFilter),
    fullyConnected :: !Layer,
    outputLayer :: !Layer
} deriving (Show, Generic)

instance NFData CNN

instance Binary.Binary Neuron where
    put (Neuron w b) = Binary.put (VU.toList w) >> Binary.put b
    get = Neuron <$> (VU.fromList <$> Binary.get) <*> Binary.get

instance Binary.Binary Layer where
    put (Layer n a) = Binary.put (V.toList n) >> Binary.put (VU.toList a)
    get = Layer <$> (V.fromList <$> Binary.get) <*> (VU.fromList <$> Binary.get)

instance Binary.Binary ConvFilter where
    put (ConvFilter w b) = Binary.put (map VU.toList $ V.toList w) >> Binary.put b
    get = ConvFilter <$> (V.fromList . map VU.fromList <$> Binary.get) <*> Binary.get

instance Binary.Binary CNN where
    put (CNN c1 c2 fc out) = do
        Binary.put (V.toList c1)
        Binary.put (V.toList c2)
        Binary.put fc
        Binary.put out
    get = CNN <$> (V.fromList <$> Binary.get) 
             <*> (V.fromList <$> Binary.get)
             <*> Binary.get
             <*> Binary.get

data MNISTData = MNISTData {
    images :: !(V.Vector (VU.Vector Double)),
    labels :: !(V.Vector (VU.Vector Double))
}

sigmoid :: Double -> Double
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

sigmoidDerivative :: Double -> Double
sigmoidDerivative !x = s * (1 - s)
  where !s = sigmoid x
{-# INLINE sigmoidDerivative #-}

convolve :: VU.Vector Double -> ConvFilter -> VU.Vector Double
convolve !input !filter = VU.generate outputSize $ \i ->
    let !x = i `div` outputWidth
        !y = i `mod` outputWidth
        !result = sum [((input VU.! (x * inputWidth + j)) * 
                       (filterWeights filter V.! 0 VU.! j))
                     | j <- [0..filterSize-1]
                     ] + filterBias filter
    in sigmoid result
  where
    !inputWidth = 28
    !filterSize = 5
    !outputWidth = inputWidth - filterSize + 1
    !outputSize = outputWidth * outputWidth
{-# INLINE convolve #-}

maxPool :: VU.Vector Double -> VU.Vector Double
maxPool !input = VU.generate outputSize $ \i ->
    let !x = i `div` outputWidth * 2
        !y = i `mod` outputWidth * 2
        !vals = [input VU.! (x * inputWidth + y),
                input VU.! (x * inputWidth + y + 1),
                input VU.! ((x + 1) * inputWidth + y),
                input VU.! ((x + 1) * inputWidth + y + 1)]
    in maximum vals
  where
    !inputWidth = 24
    !outputWidth = inputWidth `div` 2
    !outputSize = outputWidth * outputWidth
{-# INLINE maxPool #-}

flatten :: VU.Vector Double -> VU.Vector Double
flatten = id
{-# INLINE flatten #-}

layerForward :: Layer -> VU.Vector Double -> Layer
layerForward !layer !input = layer { activations = newActivations }
  where
    !newActivations = VU.generate (V.length $ neurons layer) $ \i ->
        let !neuron = neurons layer V.! i
            !sum = VU.sum $ VU.zipWith (*) (weights neuron) input
        in sigmoid (sum + bias neuron)
{-# INLINE layerForward #-}

randomWeight :: IO Double
randomWeight = randomRIO (-0.1, 0.1)
{-# INLINE randomWeight #-}

randomConvFilter :: Int -> IO ConvFilter
randomConvFilter !size = do
    !w <- VU.replicateM (size * size) randomWeight
    !b <- randomWeight
    return $! ConvFilter (V.singleton w) b

randomLayer :: Int -> Int -> IO Layer
randomLayer !numNeurons !inputSize = do
    !neurons <- V.replicateM numNeurons $ do
        !ws <- VU.replicateM inputSize randomWeight
        !b <- randomWeight
        return $! Neuron ws b
    return $! Layer neurons (VU.replicate numNeurons 0.0)

initCNN :: IO CNN
initCNN = do
    !conv1 <- V.replicateM 6 (randomConvFilter 5)
    !conv2 <- V.replicateM 16 (randomConvFilter 5)
    !fc <- randomLayer 120 400
    !output <- randomLayer 10 120
    return $! CNN conv1 conv2 fc output

forwardProp :: CNN -> VU.Vector Double -> (VU.Vector Double, CNN)
forwardProp !cnn !input =
    let !conv1Outputs = V.map (convolve input) (convLayer1 cnn)
        !conv1Output = VU.concat $ V.toList conv1Outputs
        !pooled1 = maxPool conv1Output
        !conv2Outputs = V.map (convolve pooled1) (convLayer2 cnn)
        !conv2Output = VU.concat $ V.toList conv2Outputs
        !pooled2 = maxPool conv2Output
        !flattened = flatten pooled2
        !fcOutput = layerForward (fullyConnected cnn) flattened
        !outputResult = layerForward (outputLayer cnn) (activations fcOutput)
    in (activations outputResult,
        cnn { fullyConnected = fcOutput, outputLayer = outputResult })

backprop :: LearningRate -> CNN -> VU.Vector Double -> VU.Vector Double -> CNN
backprop !lr !cnn !input !target =
    let (!output, !forwardState) = forwardProp cnn input
        !outputError = VU.zipWith (-) target output
        (!outputGrads, !fcError) = layerBackprop lr (outputLayer forwardState) outputError
        (!fcGrads, !conv2Error) = layerBackprop lr (fullyConnected forwardState) fcError
        !conv2Grads = convLayerBackprop lr (convLayer2 forwardState) conv2Error
        !conv1Grads = convLayerBackprop lr (convLayer1 forwardState) conv2Error
    in cnn {
        convLayer1 = V.zipWith (updateFilter lr) (convLayer1 cnn) conv1Grads,
        convLayer2 = V.zipWith (updateFilter lr) (convLayer2 cnn) conv2Grads,
        fullyConnected = updateLayer lr (fullyConnected cnn) fcGrads,
        outputLayer = updateLayer lr (outputLayer cnn) outputGrads
    }

layerBackprop :: LearningRate -> Layer -> VU.Vector Delta -> (V.Vector (VU.Vector Weight, Bias), VU.Vector Delta)
layerBackprop !lr !layer !deltas =
    let !gradients = V.zip 
            (V.map (\n -> VU.map (* lr) (weights n)) (neurons layer))
            (V.fromList $ replicate (V.length $ neurons layer) (lr * VU.sum deltas))
        !nextDeltas = VU.generate (VU.length $ weights $ V.head $ neurons layer) $ \i ->
            let contributions = V.map (\n -> (weights n VU.! i) * 
                                           (deltas VU.! (V.length (neurons layer) - 1))) 
                                    (neurons layer)
            in V.sum contributions * sigmoidDerivative (activations layer VU.! i)
    in (gradients, nextDeltas)
  where
    computeGradients neuron delta =
        let !weightGrads = VU.map (* delta) (weights neuron)
            !biasGrad = delta * lr
        in (weightGrads, biasGrad)

convLayerBackprop :: LearningRate -> V.Vector ConvFilter -> VU.Vector Delta -> V.Vector (VU.Vector Weight)
convLayerBackprop !lr !filters !deltas =
    V.map (\f -> VU.map (* lr) $ head $ V.toList $ filterWeights f) filters

updateFilter :: LearningRate -> ConvFilter -> VU.Vector Weight -> ConvFilter
updateFilter !lr !filter !grads =
    filter { filterWeights = V.singleton $ VU.zipWith (-) (head $ V.toList $ filterWeights filter) grads }

updateLayer :: LearningRate -> Layer -> V.Vector (VU.Vector Weight, Bias) -> Layer
updateLayer !lr !layer !grads =
    layer { neurons = V.zipWith updateNeuron (neurons layer) grads }
  where
    updateNeuron neuron (weightGrads, biasGrad) =
        neuron { weights = VU.zipWith (-) (weights neuron) weightGrads,
                bias = bias neuron - biasGrad }

loadMNISTData :: FilePath -> FilePath -> IO MNISTData
loadMNISTData !imagesPath !labelsPath = do
    !imageData <- BS.readFile imagesPath
    !labelData <- BS.readFile labelsPath
    let !images = parseIDXImages imageData
        !labels = parseIDXLabels labelData
        !oneHotLabels = V.map oneHotEncode labels
    return $! MNISTData images oneHotLabels

parseIDXImages :: BS.ByteString -> V.Vector (VU.Vector Double)
parseIDXImages !bs =
    let !header = BS.take 16 bs
        !imageData = BS.drop 16 bs
        !numImages = fromIntegral $ runGet getWord32be $ BSL.fromStrict $ BS.take 4 $ BS.drop 4 header
        !rows = 28
        !cols = 28
    in V.generate numImages $ \i ->
        VU.generate (rows * cols) $ \j ->
            fromIntegral (BS.index imageData (i * rows * cols + j)) / 255.0

parseIDXLabels :: BS.ByteString -> V.Vector Int
parseIDXLabels !bs =
    let !labelData = BS.drop 8 bs
    in V.generate (BS.length labelData) $ \i ->
        fromIntegral $ BS.index labelData i

oneHotEncode :: Int -> VU.Vector Double
oneHotEncode !label = VU.generate 10 $ \i -> if i == label then 1.0 else 0.0

trainBatch :: LearningRate -> CNN -> [(VU.Vector Double, VU.Vector Double)] -> IO CNN
trainBatch !lr !cnn !batch = do
    return $! foldl' (\acc (input, target) ->
        let !cnnUpdated = backprop lr acc input target
        in cnnUpdated) cnn batch

train :: LearningRate -> Int -> Int -> CNN -> MNISTData -> IO CNN
train !lr !epochs !batchSize !initialCNN !mnistData = do
    let !trainingPairs = V.zip (images mnistData) (labels mnistData)
        !numBatches = V.length trainingPairs `div` batchSize
        !totalSteps = epochs * numBatches
    
    foldM (\cnn epoch -> do
        printf "\nEpoch %d/%d\n" epoch epochs
        printf "["
        hFlush stdout
        
        let !batches = [V.toList $ V.slice (i * batchSize) batchSize trainingPairs 
                      | i <- [0..numBatches-1]]
        
        foldM (\cnn' (batchNum, batch) -> do
                when (batchNum `mod` (numBatches `div` 50) == 0) $ do
                    let progress = ((epoch - 1) * numBatches + batchNum) * 100 `div` totalSteps
                    printf "\rEpoch %d/%d [%s%s] %d%%" 
                        epoch epochs 
                        (replicate (progress `div` 2) '=')
                        (replicate (50 - (progress `div` 2)) ' ')
                        progress
                    hFlush stdout
                
                !trainedCNN <- trainBatch lr cnn' batch
                return $! trainedCNN
            ) cnn (zip [1..] batches)
        ) initialCNN [1..epochs]

saveCNN :: FilePath -> CNN -> IO ()
saveCNN path cnn = BSL.writeFile path (Binary.encode cnn)

loadCNN :: FilePath -> IO CNN
loadCNN path = do
    content <- BSL.readFile path
    case Binary.decode content of
        Left err -> error $ "Failed to load model: " ++ err
        Right cnn -> return cnn

processImage :: VU.Vector Double -> CNN -> (VU.Vector Double, VU.Vector Double)
processImage !input !cnn =
    let !conv1Outputs = V.map (convolve input) (convLayer1 cnn)
        !conv1Output = VU.concat $ V.toList conv1Outputs
        !pooled1 = maxPool conv1Output
        !conv2Outputs = V.map (convolve pooled1) (convLayer2 cnn)
        !conv2Output = VU.concat $ V.toList conv2Outputs
        !pooled2 = maxPool conv2Output
        (!final, _) = forwardProp cnn input
    in (conv1Output, final)

interactiveMode !cnn = do
    hSetBuffering stdin LineBuffering
    hSetBuffering stdout LineBuffering
    
    forever $ do
        input <- getLine
        case Aeson.decode (BSL.pack input) of
            Nothing -> do
                putStrLn "Error: Invalid input format"
                hFlush stdout
            
            Just imgData -> do
                let (!layerOutput, !predictions) = processImage imgData cnn
                    result = object [
                        fromString "layer_output" .= VU.toList layerOutput,
                        fromString "predictions" .= VU.toList predictions
                        ]
                
                BSL.putStr (Aeson.encode result)
                putStr "\n"
                hFlush stdout

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["train"] -> do
            putStrLn "Loading MNIST data..."
            !mnistData <- loadMNISTData "data/train-images.idx3-ubyte" 
                                      "data/train-labels.idx1-ubyte"
            
            putStrLn "Initializing CNN..."
            !cnn <- initCNN
            
            putStrLn "Starting training..."
            !trainedCNN <- train 0.005 8 64 cnn mnistData
            
            putStrLn "\nSaving model..."
            saveCNN "trained_model.cnn" trainedCNN
            
            putStrLn "Training complete!"
        
        [] -> do
            modelExists <- doesFileExist "trained_model.cnn"
            if not modelExists 
                then putStrLn "Error: Model file not found!"
                else do
                    !cnn <- loadCNN "trained_model.cnn"
                    interactiveMode cnn