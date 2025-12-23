from vllm.v1.core.cml_parser import CMLParser

def test_cml_parser():
    parser = CMLParser()
    
    # Case 1: Split Call
    text_chunks = ["Hello ", "[", "CAL", "L]", " tool_name"]
    critical_sections = []
    for chunk in text_chunks:
        in_crit, event = parser.step(chunk)
        critical_sections.append(in_crit)
    
    # [CALL] formed at chunk 3 ("L]")
    # So chunk 0: False
    # chunk 1: False ( "[")
    # chunk 2: False ( "CAL") -> buffer: "[CAL"
    # chunk 3: True ("L]") -> buffer: "[CALL]"
    print(f"Critical Sections: {critical_sections}")
    assert critical_sections == [False, False, False, True, True]

    # Case 2: End Call
    text_chunks = ["(arg)", "[", "EN", "D]"]
    for chunk in text_chunks:
        in_crit, event = parser.step(chunk)
        critical_sections.append(in_crit)
    
    # [END] formed at last chunk.
    # So chunk 0,1,2 should be True. 
    # Last chunk: True (detected) -> False (state updated).
    # Wait, if parser.step returns current state.
    # If [END] detected, it transitions to False.
    # checking logic: 
    # if self.in_call: check END. if found -> in_call = False.
    # So the return value for the chunk containing [END] will be False.
    print(f"End Sections: {critical_sections[5:]}")
    assert critical_sections[5:] == [True, True, True, False]
    
    # Case 3: Trap
    parser.reset()
    in_crit, event = parser.step("Some text [TR")
    assert event is None
    in_crit, event = parser.step("AP]")
    assert event == "TRAP"
    print("Trap detected.")

if __name__ == "__main__":
    test_cml_parser()
