#mcandrew;

def timestamp():
    from datetime import datetime
    time = datetime.now()
    return "{:04d}-{:02d}-{:02d}-{:02d}".format(time.year,time.month,time.day,time.hour)
