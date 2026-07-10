rates = {
    "USD": 1,
    "EUR": 0.85,
    "INR": 83.5,
    "GBP": 0.74,
    "JPY": 146.2
}

try:
    amount = float(input("Enter amount: "))
    from_currency = input("From currency: ").upper()
    to_currency = input("To currency: ").upper()

    if from_currency not in rates or to_currency not in rates:
        print("Invalid currency code!")
    else:
        result = amount * rates[to_currency] / rates[from_currency]
        print(f"Converted Amount: {result:.2f} {to_currency}")

except ValueError:
    print("Please enter a valid number.")
