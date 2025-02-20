
import itpp
# /provides tools for communication systems, including channel coding 
# (e.g., Hamming codes), modulation, demodulation, and signal processing.

#this function calculates the BLER(Block error ratio) for a hamming coded
#system over an AWGN (Additive White Gaussian Noise) channel
'''
BLER:
fraction of data blocks (or frames) that are incorrectly 
received after decoding.
block: A group of bits transmitted together as a unit. in 7,4 hamming code it has
7 bits
bler=no.of blocks with errors/total number of transmitted blocks
AWGN : a mathematical model to represent noise in comms channel
additive? noise is additive
white? noise power is uniformly distributed across all freq
just like white light containg all colours
gaussian: amp of the noise folllows a gaussian/normal distribution
this noise has mean=0, variance= channel's signal to noise ratio
S-N-R: signal to noise ratio quantifies the strength of the transmitted signal
relative to the noise:
signal power/noise power
jitna zyada utna badiya hai 
why awgn? standard noise model and simplify plus real world simulation is almost 
same, relevant to systems operating wireless
HAMMING CODING:
error correction
detect and correctsingle bit errorrs in transmitted data
(n,k)-> n= total no. of bits (info+redundancy). k= no. of info bits
(7,4) 4 bits contain info and 3 contain parity for error  correction
decoding:
codeword is checked for errors by a syndrome decoding method if a single bit error is 
detectedthe corrupted bit is identified and corrected 
'''
def block_error_ratio_hamming_awgn(snr_db, block_size):
    #snr in  decibel
    #block_size=4, for(7,4)hc
    
     # Mapping from k (block size) to m. m = 3 implies (7,4) code (a dictionary)
    mapping_k_m = {4: 3}
    m = mapping_k_m[block_size]
     
    '''Hamming encoder and decoder instance'''
    hamm = itpp.comm.Hamming_Code(m)
    # hamm is an encoder and decoder for the hamming code
    n = pow(2,m) - 1 # channel use
    rate = float(block_size)/float(n)
    #rate= info bits/total bits
    
    '''Generate random bits'''
    nrof_bits = 10000 * block_size
    source_bits = itpp.randb(nrof_bits)
    #generates a vector of random bita
    
    '''Encode the bits'''
    encoded_bits = hamm.encode(source_bits)
    
    #BPSK MODULATION
    '''
    it stands for Binary phase key shifting:simple digital modulation in comms
    modulation: digital to  analogue to transmit over a physical medium like air
    or wires
    each binary bit is mapped to one of two phases of a carrirer wave 
    binary 0 is phase 0 degree
    binary 1 is phase 180 degree
    the phase is of carrier wave
    so, the carrier wave flips its phase dpeending upon the input bit
    MATHEMATICALLY:
    s(t)=Acos(2*pi*fc*t+theta);
    Constellation diagram  in BPSK
    ONLY 2 pts on the real axis
    +1+0i for binary 1;
    -1+0i for binary 0
    this allows reciever only to differentiate between two phases
    adv: simple,works well with noise, energy efficient
    disad: low data rate 1 bitpersymbol slower than QPSK, 16-QAM
    '''
    '''Modulate the bits:
    this section performs BPSK modulation, converting the encoded binary bits
    into the modulated symbols (waveform representation for transmission)
    each bit is mapped to a constellation point in the complex plane
    '''
    # 2d modulator, allows us to define a constellation (a set of symbols)
    #in the complex plane map input bits to these (2d is not req here as only 0&1)
    modulator_ = itpp.comm.modulator_2d()
    #constell: set a pts in complex plane representinf the symbols for modulation
    #all the is mapping the pts with bits
    constellation = itpp.cvec('-1+0i, 1+0i')
    symbols = itpp.ivec('0, 1')
    modulator_.set(constellation, symbols)
    ''' 0,1,1,0->[−1+0i,1+0i,1+0i,−1+0i].'''
    tx_signal = modulator_.modulate_bits(encoded_bits)
    
    '''Add the effect of channel to the signal'''
    noise_variance = 1.0 / (rate * pow(10, 0.1 * snr_db))
    noise = itpp.randn_c(tx_signal.length())
    noise *= itpp.math.sqrt(noise_variance)
    rx_signal = tx_signal + noise
    
    '''Demodulate the signal:
    method converts the received signal back into binary 
    bits by mapping symbols to the nearest constellation points.'''
    demodulated_bits = modulator_.demodulate_bits(rx_signal)
    
    '''Decode the received bits:  attempts to correct errors in the
    demodulated bits using the Hamming decoder.
    '''
    decoded_bits = hamm.decode(demodulated_bits) 
    
    '''Calculate the block error ratio'''
    #initialises abler counter  
    blerc = itpp.comm.BLERC(block_size)
    #cnts
    blerc.count(source_bits, decoded_bits)
    #returns the fraction of blocks incorrectly decoded
    return blerc.get_errorrate()