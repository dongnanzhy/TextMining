import os.path,subprocess
import re
import sys
import fileinput
import pagerank

def writeFile(text):
    with open("Exp-Result.txt", "a") as myfile:
        myfile.write(text+'\n')
def weigthTuning():
    replacement = []
    for weight in range(100, 0, -1):
        replacement.append('weight:'+str(weight/100.0))
    return replacement
def dampingTuning():
    replacement = []
    for beta in range(1, 10, 1):
        replacement.append('beta:'+str(beta/10.0))
    return replacement

MAPPattern = re.compile(r'^(?i)MAP\s+\d+')
MAPAllPattern = re.compile(r'^(?i)MAP.*all.*')
P00AllPattern = re.compile(r'^(?i)ircl_prn.0.00\s.*all.*')
P01AllPattern = re.compile(r'^(?i)ircl_prn.0.10\s.*all.*')
P02AllPattern = re.compile(r'^(?i)ircl_prn.0.20\s.*all.*')
P03AllPattern = re.compile(r'^(?i)ircl_prn.0.30\s.*all.*')
P04AllPattern = re.compile(r'^(?i)ircl_prn.0.40\s.*all.*')
P05AllPattern = re.compile(r'^(?i)ircl_prn.0.50\s.*all.*')
P06AllPattern = re.compile(r'^(?i)ircl_prn.0.60\s.*all.*')
P07AllPattern = re.compile(r'^(?i)ircl_prn.0.70\s.*all.*')
P08AllPattern = re.compile(r'^(?i)ircl_prn.0.80\s.*all.*')
P09AllPattern = re.compile(r'^(?i)ircl_prn.0.90\s.*all.*')
P10AllPattern = re.compile(r'^(?i)ircl_prn.1.00\s.*all.*')

initial = 'weight:1.0'
replacement = weigthTuning()
# initial = 'beta:0.1'
# replacement = dampingTuning()


if os.path.isfile("Exp-Result.txt"):
    os.remove("Exp-Result.txt")

mapMax = 0.0
for r in replacement:
    parameter = fileinput.input('parameter.txt', inplace=1)
    for i, line in enumerate(parameter):
        sys.stdout.write(line.replace(initial, r))
    print r
    parameter.close()
    initial = r
    parameter = open('parameter.txt', 'r')
    writeFile(parameter.read())
    parameter.close()

    pagerank.run()

    result = os.popen('''
        perl -e '
            use LWP::Simple;
            my $fileIn = "../data/result.txt";
            my $url = "http://nyc.lti.cs.cmu.edu/classes/11-741/s16/HW/HW3/upload.cgi";
            my $ua = LWP::UserAgent->new();
            my $result = $ua->post($url,
                   Content_Type => "form-data",
                   Content => [ logtype => "Summary", infile => [$fileIn] ]);
            my $result = $result->as_string;
            $result =~ s/<BR>/\n/g;
            print $result;'
        ''').read()
    result = result.splitlines()
    mapAll = 0.0
    for each in result:
        if MAPAllPattern.match(each):
            split = each.split()
            mapAll = split[2]
        elif P00AllPattern.match(each):
            split = each.split()
            writeFile("0%: " + split[2])
        elif P01AllPattern.match(each):
            split = each.split()
            writeFile("10%: " + split[2])
        elif P02AllPattern.match(each):
            split = each.split()
            writeFile("20%: " + split[2])
        elif P03AllPattern.match(each):
            split = each.split()
            writeFile("30%: " + split[2])
        elif P04AllPattern.match(each):
            split = each.split()
            writeFile("40%: " + split[2])
        elif P05AllPattern.match(each):
            split = each.split()
            writeFile("50%: " + split[2])
        elif P06AllPattern.match(each):
            split = each.split()
            writeFile("60%: " + split[2])
        elif P07AllPattern.match(each):
            split = each.split()
            writeFile("70%: " + split[2])
        elif P08AllPattern.match(each):
            writeFile("80%: " + split[2])
        elif P09AllPattern.match(each):
            split = each.split()
            writeFile("90%: " + split[2])
        elif P10AllPattern.match(each):
            split = each.split()
            writeFile("100%: " + split[2])
    if mapAll > mapMax:
        mapMax = mapAll
    print "MAPALL@"+r+": "+str(mapAll)
    writeFile("MAPALL: " + str(mapAll)+ "\n")
print "Max MAP: "+str(mapMax)
