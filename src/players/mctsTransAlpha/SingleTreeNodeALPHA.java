package players.mctsTransAlpha;

import core.GameState;
import players.heuristics.AdvancedHeuristic;
import players.heuristics.CustomHeuristic;
import players.heuristics.StateHeuristic;
import utils.ElapsedCpuTimer;
import utils.Types;
import utils.Utils;
import utils.Vector2d;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Random;

import static java.lang.Math.max;

public class SingleTreeNodeALPHA
{
    public MCTSParamsALPHA params;
    private double paramVal = 0.4;

    private final SingleTreeNodeALPHA parent;
    private final SingleTreeNodeALPHA[] children;
    private final Random m_rnd;
    private final int m_depth;
    private final int childIdx;
    private int stateIdx;
    private int fmCallsCount;

    private final int num_actions;
    private final Types.ACTIONS[] actions;

    private GameState rootState;
    private StateHeuristic rootStateHeuristic;

    // Transpositions table
    private static final Hashtable statesTable = new Hashtable();

    SingleTreeNodeALPHA(MCTSParamsALPHA p, Random rnd, int num_actions, Types.ACTIONS[] actions) {
        this(p, null, -1, rnd, num_actions, actions, 0, null);
    }

    private SingleTreeNodeALPHA(MCTSParamsALPHA p, SingleTreeNodeALPHA parent, int childIdx, Random rnd, int num_actions,
                                Types.ACTIONS[] actions, int fmCallsCount, StateHeuristic sh) {
        this.params = p;
        this.fmCallsCount = fmCallsCount;
        this.parent = parent;
        this.m_rnd = rnd;
        this.num_actions = num_actions;
        this.actions = actions;
        children = new SingleTreeNodeALPHA[num_actions];
        this.childIdx = childIdx;
        if(parent != null) {
            m_depth = parent.m_depth + 1;
            this.rootStateHeuristic = sh;
        }
        else
            m_depth = 0;
    }

    // Object that holds information of the stat
    // Will be placed in the hashtable
    class StateInfo{
        int nVisits;
        double[][] actionInfo;
        double[][] amafInfo;
        double[] bounds;

        public StateInfo(){
            nVisits = 0;
            actionInfo = new double[6][2];
            for (double[] i : actionInfo) {
                i = new double[]{0,0};
            }
            amafInfo = new double[6][2];
            for (double[] i : amafInfo) {
                i = new double[]{0,0};
            }

            bounds = new double[]{Double.MAX_VALUE, -Double.MAX_VALUE};
        }
    }

    void setRootGameState(GameState gs)
    {
        this.rootState = gs;
        if (params.heuristic_method == params.CUSTOM_HEURISTIC)
            this.rootStateHeuristic = new CustomHeuristic(gs);
        else if (params.heuristic_method == params.ADVANCED_HEURISTIC) // New method: combined heuristics
            this.rootStateHeuristic = new AdvancedHeuristic(gs, m_rnd);

        statesTable.clear();
        int rootStateIdx = gs.toString().hashCode();
        statesTable.put(rootStateIdx, new StateInfo());
        this.stateIdx = rootStateIdx;
    }


    void mctsSearch(ElapsedCpuTimer elapsedTimer) {

        double avgTimeTaken;
        double acumTimeTaken = 0;
        long remaining;
        int numIters = 0;

        int remainingLimit = 5;
        boolean stop = false;

        while(!stop){

            GameState state = rootState.copy();
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();
            // Selection Step
            SingleTreeNodeALPHA selected = treePolicy(state);
            // Simulation Step
            Object[] delta = selected.rollOut(state);
            // Backprobagation Step
            backUp(selected, (double) delta[0], (ArrayList<Integer>) delta[1]);

            //Stopping condition
            if(params.stop_type == params.STOP_TIME) {
                numIters++;
                acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;
                avgTimeTaken  = acumTimeTaken/numIters;
                remaining = elapsedTimer.remainingTimeMillis();
                stop = remaining <= 2 * avgTimeTaken || remaining <= remainingLimit;
            }else if(params.stop_type == params.STOP_ITERATIONS) {
                numIters++;
                stop = numIters >= params.num_iterations;
            }else if(params.stop_type == params.STOP_FMCALLS)
            {
                fmCallsCount+=params.rollout_depth;
                stop = (fmCallsCount + params.rollout_depth) > params.num_fmcalls;
            }
        }
        //System.out.println(" ITERS " + numIters);
    }

    private SingleTreeNodeALPHA treePolicy(GameState state) {

        SingleTreeNodeALPHA cur = this;

        while (!state.isTerminal() && cur.m_depth < params.rollout_depth)
        {
            // Perform Selection Step
            // Check if current node being explored is a leaf node
            if (cur.notFullyExpanded()) {
                return cur.expand(state);
            }
            // If it is not a leaf node perform UCT to select a child node
            else {
                cur = cur.uct(state);
            }
        }

        return cur;
    }

    // Expand the tree node by selecting the next available action
    private SingleTreeNodeALPHA expand(GameState state) {

        int bestAction = 0;
        double bestValue = -1;

        for (int i = 0; i < children.length; i++) {
            double x = m_rnd.nextDouble();
            if (x > bestValue && children[i] == null) {
                bestAction = i;
                bestValue = x;
            }
        }

        //Roll the state
        roll(state, actions[bestAction]);

        int stateID = state.toString().hashCode();
        if (!statesTable.containsKey(stateID))
            statesTable.put(stateID, new StateInfo());

        SingleTreeNodeALPHA tn = new SingleTreeNodeALPHA(params,this,bestAction,this.m_rnd,num_actions,
                actions, fmCallsCount, rootStateHeuristic);
        tn.stateIdx = stateID;
        children[bestAction] = tn;
        return tn;
    }

    // Roll the state forward
    private void roll(GameState gs, Types.ACTIONS act)
    {
        //Simple, all random first, then my position.
        int nPlayers = 4;
        Types.ACTIONS[] actionsAll = new Types.ACTIONS[4];
        int playerId = gs.getPlayerId() - Types.TILETYPE.AGENT0.getKey();

        for(int i = 0; i < nPlayers; ++i)
        {
            if(playerId == i)
            {
                actionsAll[i] = act;
            }else {
                int actionIdx = m_rnd.nextInt(gs.nActions());
                actionsAll[i] = Types.ACTIONS.all().get(actionIdx);
            }
        }

        gs.next(actionsAll);

    }

    // Select the best child node using Upper Confidence Bound
    private SingleTreeNodeALPHA uct(GameState state) {
        SingleTreeNodeALPHA selected = null;
        double bestValue = -Double.MAX_VALUE;

        // Get the node state information from the hashtable
        StateInfo info = (StateInfo) statesTable.get(this.stateIdx);
        int i = 0;
        for (double[] action : info.actionInfo)
        {
            // Child_Value is the standard Q value (totValue of node / number of visits to the node)
            // Get the total reward for the action and the N(s,a)
            // and calculate the standard Q(s,a)
            double hvVal = action[1];
            double childValue =  hvVal / (action[0] + params.epsilon);

            // amadChildVal is the amafscore for the action
            // Get the total reward and the N(s,a) gotten for the amaf score
            // compute the amaf score
            double amafHvVal = info.amafInfo[i][1];
            double amafChildVal = amafHvVal / (info.amafInfo[i][0] + params.epsilon);

            // The star child is the Q*(s,a) obtained by combining the amaf score and the standard score
            // obtained using the RAVE technique
            double starChild = (paramVal * amafChildVal) + ((1 - paramVal) * childValue);

            starChild = Utils.normalise(starChild, info.bounds[0], info.bounds[1]);

            // UCT_value is the decision value (Q-value + C*srt(ln(number of visits) / number of times action has been taken from state))
            double uctValue = starChild +
                    params.K * Math.sqrt(Math.log(info.nVisits + 1) / (action[0] + params.epsilon));

            uctValue = Utils.noise(uctValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly

            // small sampleRandom numbers: break ties in unexpanded nodes
            if (uctValue > bestValue) {
                selected = this.children[i];
                bestValue = uctValue;
            }

            i++;
        }
        if (selected == null)
        {
            throw new RuntimeException("Warning! returning null: " + bestValue + " : " + this.children.length + " " +
                    + info.bounds[0] + " " + info.bounds[1]);
        }

        //Roll the state:
        roll(state, actions[selected.childIdx]);

        return selected;
    }

    //
    private Object[] rollOut(GameState state)
    {
        int thisDepth = this.m_depth;
        ArrayList<Integer> actionsTaken = new ArrayList<Integer>();

        while (!finishRollout(state,thisDepth)) {
            int action = safeRandomAction(state);
            actionsTaken.add(action);
            roll(state, actions[action]);
            thisDepth++;
        }

        double result = rootStateHeuristic.evaluateState(state);

        return new Object[]{result, actionsTaken};
    }

    private int safeRandomAction(GameState state)
    {
        Types.TILETYPE[][] board = state.getBoard();
        ArrayList<Types.ACTIONS> actionsToTry = Types.ACTIONS.all();
        int width = board.length;
        int height = board[0].length;

        while(actionsToTry.size() > 0) {

            int nAction = m_rnd.nextInt(actionsToTry.size());
            Types.ACTIONS act = actionsToTry.get(nAction);
            Vector2d dir = act.getDirection().toVec();

            Vector2d pos = state.getPosition();
            int x = pos.x + dir.x;
            int y = pos.y + dir.y;

            if (x >= 0 && x < width && y >= 0 && y < height)
                if(board[y][x] != Types.TILETYPE.FLAMES)
                    return nAction;

            actionsToTry.remove(nAction);
        }

        //Uh oh...
        return m_rnd.nextInt(num_actions);
    }

    @SuppressWarnings("RedundantIfStatement")
    private boolean finishRollout(GameState rollerState, int depth)
    {
        if (depth >= params.rollout_depth)      //rollout end condition.
            return true;

        if (rollerState.isTerminal())               //end of game
            return true;

        return false;
    }

    private void backUp(SingleTreeNodeALPHA node, double result, ArrayList<Integer> rolloutActions)
    {
        SingleTreeNodeALPHA n = node;

        // Update AMAF score for the actions taken in the rollout
        for (int actionIdx : rolloutActions) {
            n = n.parent;
            while (n != null){
                StateInfo info = (StateInfo) statesTable.get(n.stateIdx);
                info.amafInfo[actionIdx][0]++;
                info.amafInfo[actionIdx][1] += result;

                n = n.parent;
            }
            n = node;
        }

        // Update standard scores and AMAF scores for the actions taken by the Selection step
        n = node;
        while(n != null)
        {
            StateInfo info = (StateInfo) statesTable.get(n.stateIdx);
            info.nVisits++;

            if (result < info.bounds[0]) {
                info.bounds[0] = result;
            }
            if (result > info.bounds[1]) {
                info.bounds[1] = result;
            }

            // update the standard score of the parent node of the node currently being analysed
            SingleTreeNodeALPHA np = n.parent;
            if(np != null) {
                info = (StateInfo) statesTable.get(np.stateIdx);
                info.actionInfo[n.childIdx][0]++;
                info.actionInfo[n.childIdx][1] += result;

                // update the amaf scores for the current action as if it was taken in previous steps
                // (Updates the amaf scores for each node above the parent node of the current node being analysed)
                SingleTreeNodeALPHA npa = np.parent;
                while(npa != null){
                    info = (StateInfo) statesTable.get(npa.stateIdx);
                    info.amafInfo[n.childIdx][0]++;
                    info.amafInfo[n.childIdx][1] += result;

                    npa = npa.parent;
                }
            }

            n = np;
        }
    }


    int mostVisitedAction() {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;
        boolean allEqual = true;
        double first = -1;

        StateInfo info = (StateInfo) statesTable.get(stateIdx);
        for (int i=0; i<info.actionInfo.length; i++) {

            if(info.actionInfo[i] != null)
            {
                if(first == -1)
                    first = info.actionInfo[i][0];
                else if(first != info.actionInfo[i][0])
                {
                    allEqual = false;
                }

                double childValue = info.actionInfo[i][0];
                childValue = Utils.noise(childValue, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            selected = 0;
        }else if(allEqual)
        {
            //If all are equal, we opt to choose for the one with the best Q.
            selected = bestAction();
        }

        return selected;
    }

    int bestAction()
    {
        int selected = -1;
        double bestValue = -Double.MAX_VALUE;

        StateInfo info = (StateInfo) statesTable.get(stateIdx);
        for (int i=0; i<info.actionInfo.length; i++) {

            if(info.actionInfo[i] != null) {
                double childValue = info.actionInfo[i][1] / (info.actionInfo[i][0] + params.epsilon);
                double amafChildVal = info.amafInfo[i][1] / (info.amafInfo[i][0] + params.epsilon);
                double starChild = (paramVal * amafChildVal) + ((1 - paramVal) * childValue);
                childValue = Utils.noise(starChild, params.epsilon, this.m_rnd.nextDouble());     //break ties randomly
                if (childValue > bestValue) {
                    bestValue = childValue;
                    selected = i;
                }
            }
        }

        if (selected == -1)
        {
            System.out.println("Unexpected selection!");
            selected = 0;
        }

        return selected;
    }


    private boolean notFullyExpanded() {
        for (SingleTreeNodeALPHA tn : children) {
            if (tn == null) {
                return true;
            }
        }

        return false;
    }
}
