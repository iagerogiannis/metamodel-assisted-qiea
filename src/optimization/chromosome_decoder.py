class ChromosomeDecoder:
    def __init__(self, design_variables, encoding='binary'):
        self.design_variables = design_variables
        self.encoding = encoding

    def decode(self, chromosome):
        def decode_variable(chromosome, lower_bound, upper_bound):
            binary_chromosome = self.gray_to_binary(chromosome) if self.encoding == 'gray' else chromosome
            x = sum([binary_chromosome[i] * 2 ** (len(binary_chromosome) - 1 - i) for i in range(len(binary_chromosome))])
            decoded =  lower_bound + x * (upper_bound - lower_bound) / (2 ** len(binary_chromosome) - 1)
            return decoded

        decoded = []
        index = 0
        for variable in self.design_variables:
            num_of_bits = variable['bits']
            lower_bound = variable['lower_bound']
            upper_bound = variable['upper_bound']
            variable_chromosome = chromosome[index:index + num_of_bits]
            decoded.append(decode_variable(variable_chromosome, lower_bound, upper_bound))
            index += num_of_bits

        return decoded

    @staticmethod 
    def gray_to_binary(gray_code):
        binary_code = [gray_code[0]]

        for i in range(1, len(gray_code)):
            binary_code.append(binary_code[i-1] ^ gray_code[i])

        return binary_code
