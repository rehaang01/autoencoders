
import itpp

def block_error_ratio_uncoded_awgn(snr_db, block_size):
    #function to calculate the block error ratio for 
    # #uncoded data transmitted over an AWGN channel.
    '''Generate random bits'''
    nrof_bits = 3 * 10000 * block_size
    #large number of bits for accurate eror 
    source_bits = itpp.randb(nrof_bits)
    #as all the bits are info bits, so the rate is 1
    rate = 1.0
    
    '''Modulate the bits'''
    modulator_ = itpp.comm.modulator_2d()
    constellation = itpp.cvec('-1+0i, 1+0i')
    symbols = itpp.ivec('0, 1')
    modulator_.set(constellation, symbols)
    tx_signal = modulator_.modulate_bits(source_bits)
    
    '''Add the effect of channel to the signal'''
    noise_variance = 1.0 / (rate * pow(10, 0.1 * snr_db))
    noise = itpp.randn_c(tx_signal.length())
    noise *= itpp.math.sqrt(noise_variance)
    rx_signal = tx_signal + noise
    
    '''Demodulate the signal:
    Extracts binary bits from the noisy received signal using
    the same constellation mapping defined earlier.
    '''
    demodulated_bits = modulator_.demodulate_bits(rx_signal)
    
    '''Calculate the block error ratio'''
    blerc = itpp.comm.BLERC(block_size)
    blerc.count(source_bits, demodulated_bits)
    return blerc.get_errorrate()